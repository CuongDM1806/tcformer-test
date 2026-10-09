"""T-TIME and BFT test-time adaptation, from the authors' DeepTransferEEG.

Source: https://github.com/sylyoung/DeepTransferEEG (commit 603e30d, MIT
License, Copyright (c) 2023 siyangli).

* T-TIME: S. Li, Z. Wang, H. Luo, L. Ding, D. Wu, "T-TIME: Test-Time
  Information Maximization Ensemble for Plug-and-Play BCIs", IEEE TBME, 2024
  (tl/ttime.py and tl/ttime_ensemble.py).
* BFT: "Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based
  BCIs", IEEE JBHI, 2026 (tl/bft.py).

The backbone (EEGNet_feature + FC_xy), Euclidean alignment, the online
adaptation loops and the SML ensemble are copied from those files. The only
changes are interface changes for this repository: the target stream is passed
as an EEG array without labels (labels are used afterwards, by the caller, only
for scoring), the class-imbalanced branches are omitted (all evaluations here
are balanced), and tensors are moved to ``device`` instead of reading
``args.data_env``. Every numerical step is unchanged.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import fractional_matrix_power
from scipy.signal import hilbert


# --------------------------------------------------------------------------
# tl/models/EEGNet.py :: EEGNet_feature, tl/models/FC.py :: FC_xy
# --------------------------------------------------------------------------
class EEGNet_feature(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: float,
                 norm_rate: float):
        super(EEGNet_feature, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        return output


class FC_xy(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_xy, self).__init__()
        self.nn_out = nn_out
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        y = self.fc(x)
        return x, y


def backbone_net(chn, time_sample_num, sample_rate, class_num):
    """tl/utils/network.py :: backbone_net(args, return_type='xy')."""
    netF = EEGNet_feature(n_classes=class_num,
                          Chans=chn,
                          Samples=time_sample_num,
                          kernLenght=int(sample_rate // 2),
                          F1=4,
                          D=2,
                          F2=8,
                          dropoutRate=0.25,
                          norm_rate=0.5)
    feature_deep_dim = 8 * (time_sample_num // 32)
    netC = FC_xy(feature_deep_dim, class_num)
    return netF, netC, feature_deep_dim


# --------------------------------------------------------------------------
# tl/utils/alg_utils.py :: EA, EA_online; tl/utils/loss.py :: Entropy
# --------------------------------------------------------------------------
def EA(x):
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def EA_reference(x):
    """The R^{-1/2} that EA(x) applies, so it can be reused on later sessions."""
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    return fractional_matrix_power(np.mean(cov, 0), -0.5)


def EA_online(x, R, sample_num):
    cov = np.cov(x)
    refEA = (R * sample_num + cov) / (sample_num + 1)
    return refEA


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def _as_real(matrix):
    # fractional_matrix_power returns a complex array when round-off produces
    # tiny negative eigenvalues; np.dot with the real EEG then keeps the dtype.
    # The authors' code casts the result to float32 through torch, which keeps
    # the real part, so do the same explicitly.
    return np.real(matrix)


# --------------------------------------------------------------------------
# tl/ttime.py :: TTIME (balanced branch)
# --------------------------------------------------------------------------
def TTIME(X_stream, model, args, device):
    """Online T-TIME adaptation of one model over an unlabeled target stream.

    X_stream: (n_trials, chn, time_sample_num) target EEG in arrival order.
    Returns the per-trial softmax predictions made *before* each update.
    """
    y_pred = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0

    # loop through test data stream one by one
    for i in range(len(X_stream)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        inputs = torch.from_numpy(X_stream[i]).to(torch.float32)
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).cpu()

        # accumulate test data
        if i == 0:
            data_cum = inputs.float().cpu()
        else:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)

        # Incremental EA
        if args.align:
            if i == 0:
                sample_test = data_cum.reshape(args.chn, args.time_sample_num)
            else:
                sample_test = data_cum[i].reshape(args.chn, args.time_sample_num)
            # update reference matrix
            R = EA_online(sample_test, R, i)

            sqrtRefEA = _as_real(fractional_matrix_power(R, -0.5))
            # transform current test sample
            sample_test = np.dot(sqrtRefEA, sample_test)
            sample_test = sample_test.reshape(1, 1, args.chn, args.time_sample_num)
        else:
            sample_test = data_cum[i].numpy()
            sample_test = sample_test.reshape(1, 1, sample_test.shape[1], sample_test.shape[2])

        sample_test = torch.from_numpy(sample_test).to(torch.float32).to(device)

        _, outputs = model(sample_test)

        softmax_out = nn.Softmax(dim=1)(outputs)
        y_pred.append(softmax_out.detach().cpu().numpy())

        #################### Phase 2: target model update ####################
        model.train()
        # sliding batch
        if (i + 1) >= args.test_batch and (i + 1) % args.stride == 0:
            if args.align:
                batch_test = np.copy(data_cum[i - args.test_batch + 1:i + 1])
                # transform test batch
                batch_test = np.dot(sqrtRefEA, batch_test)
                batch_test = np.transpose(batch_test, (1, 2, 0, 3))
            else:
                batch_test = data_cum[i - args.test_batch + 1:i + 1].numpy()
                batch_test = batch_test.reshape(args.test_batch, 1, batch_test.shape[2], batch_test.shape[3])

            batch_test = torch.from_numpy(batch_test).to(torch.float32).to(device)

            for step in range(args.steps):

                _, outputs = model(batch_test)
                outputs = outputs.float().cpu()

                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                # Conditional Entropy Minimization loss
                CEM_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                # Marginal Distribution Regularization loss
                MDR_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
                loss = CEM_loss + MDR_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

    return np.concatenate(y_pred, axis=0)


# --------------------------------------------------------------------------
# tl/ttime_ensemble.py :: SML_multiclass and the online SML loop of
# multiclass_classification()
# --------------------------------------------------------------------------
def SML_multiclass(preds, n_classes):
    # corrected implementation using one-versus-rest method, differs from paper description
    preds = np.argmax(preds, -1)

    preds_one_hot = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds_one_hot = np.stack(preds_one_hot)

    preds = preds_one_hot

    weights_all = []
    class_num = preds.shape[-1]
    for i in range(class_num):

        # {-1, 1}
        pred = np.ones((preds.shape[0], preds.shape[1])) * -1
        argmax_inds = np.argmax(preds, axis=-1)
        for j in range(len(preds)):
            for n in range(len(preds[1])):
                if argmax_inds[j, n] == i:
                    pred[j, n] = 1
                else:
                    pred[j, n] = -1
        mu = np.mean(pred, axis=1)
        deviations = pred - mu[:, np.newaxis]
        # Calculate the covariance matrix
        Q = np.dot(deviations, deviations.T) / (pred.shape[1] - 1)
        # Principal eigenvector
        v = np.linalg.eig(Q)[1][:, 0]
        if v[0] <= 0:
            v = -v
        weights = v / np.sum(v)  # ensemble weights
        weights_all.append(weights)

    weights_final = np.sum(np.array(weights_all), axis=0)

    predictions = np.einsum('a,abc->bc', weights_final, preds_one_hot)
    pred = np.argmax(predictions, axis=1)

    return pred


def SML_online_ensemble(pred, class_num):
    """pred: (num_models, num_test_samples, num_classes) online predictions."""
    ens_num = pred.shape[0]
    ens_prediction = []
    for sample in range(pred.shape[1]):
        if sample < ens_num:
            ens_pred = np.average(pred[:, sample, :], axis=0)
            curr_pred = np.argmax(ens_pred, axis=-1)
        else:
            curr_table = pred[:, :sample + 1, :]
            curr_pred = SML_multiclass(curr_table, class_num)[-1]
        ens_prediction.append(curr_pred)
    # np.linalg.eig may return complex eigenvectors; argmax of the real
    # weighted vote is what the authors' accuracy_score effectively scores.
    return np.real(np.array(ens_prediction)).astype(int)


# --------------------------------------------------------------------------
# tl/bft.py :: ReliabilityRanker, soft_rank, soft_spearman_loss,
# feature_mask_branches, freq_shift, augment_branches, train_ranker, BFT_func
# --------------------------------------------------------------------------
class ReliabilityRanker(nn.Module):
    def __init__(self, feature_dim):
        super(ReliabilityRanker, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4), nn.ELU(),
            nn.Linear(feature_dim // 4, 1))

    def forward(self, x):
        return self.net(x)


def soft_rank(x, tau=1.0):
    diff = (x.unsqueeze(1) - x.unsqueeze(0)) / tau
    return torch.sigmoid(diff).sum(dim=1)


def soft_spearman_loss(pred, target, tau=1.0):
    rp = soft_rank(pred, tau)
    rt = soft_rank(target, tau)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    corr = (rp * rt).sum() / (torch.sqrt((rp ** 2).sum()) * torch.sqrt((rt ** 2).sum()) + 1e-8)
    return 1.0 - corr


def feature_mask_branches(feat, K):
    B, D = feat.shape
    branches = []
    for k in range(K):
        masked = feat.clone()
        start, end = int(k / K * D), int((k + 1) / K * D)
        masked[:, start:end] = 0.0
        branches.append(masked)
    return branches


def freq_shift(x, f_shift, sample_rate):
    device = x.device
    arr = x.detach().cpu().numpy()
    B, _, C, T = arr.shape
    n = 1
    while n < T:
        n *= 2
    t = np.arange(n)
    shift_func = np.exp(2j * np.pi * f_shift * (1.0 / sample_rate) * t)
    out = np.zeros_like(arr)
    for b in range(B):
        for c in range(C):
            padded = np.zeros(n)
            padded[:T] = arr[b, 0, c, :]
            out[b, 0, c, :] = (hilbert(padded) * shift_func)[:T].real
    return torch.tensor(out, dtype=torch.float32, device=device)


def augment_branches(x, args):
    branches = [x]                                             # identity
    branches.append(x + (torch.rand_like(x) - 0.5) * x.std() / 2.0)  # uniform noise
    for m in (0.1, -0.1, -0.2):                               # multiplicative scaling
        branches.append(x * (1 - m))
    branches.append(freq_shift(x, 0.2, args.sample_rate))     # frequency shift up
    branches.append(freq_shift(x, -0.2, args.sample_rate))    # frequency shift down
    step = max(1, int(0.2 * args.sample_rate))                # temporal shifts
    for no in (1, 2, 3, 4, 5):
        branches.append(torch.roll(x, shifts=step * no, dims=-1))
    return branches                                           # K = 12


def train_ranker(netF, netC, ranker, loader, args, device):
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ranker.parameters(), lr=args.ranker_lr)
    netF.eval()
    netC.eval()
    ranker.train()

    max_iter = args.ranker_epoch * len(loader)
    iter_num = 0
    iter_source = iter(loader)
    while iter_num < max_iter:
        try:
            inputs, labels = next(iter_source)
        except StopIteration:
            iter_source = iter(loader)
            inputs, labels = next(iter_source)
        if inputs.size(0) <= 1:
            continue
        iter_num += 1
        inputs, labels = inputs.to(device), labels.to(device)

        real_losses, pred_losses = [], []
        if args.variant == 'BFT-A':
            views = augment_branches(inputs, args)
            for v in views:
                with torch.no_grad():
                    feat = netF(v)
                    _, logits = netC(feat)
                    real_losses.append(ce(logits, labels).item())
                pred_losses.append(ranker(feat).mean())
        else:
            with torch.no_grad():
                base_feat = netF(inputs)
            for masked in feature_mask_branches(base_feat, args.K):
                with torch.no_grad():
                    _, logits = netC(masked)
                    real_losses.append(ce(logits, labels).item())
                pred_losses.append(ranker(masked).mean())

        real_losses = torch.tensor(real_losses).to(device)
        target = F.softmax(-real_losses, dim=0)
        pred = F.softmax(-torch.stack(pred_losses).squeeze(), dim=0)

        loss = soft_spearman_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ranker.eval()


def BFT_func(X_stream, model, ranker, args, device):
    """Online, backpropagation-free BFT over an unlabeled target stream."""
    netF, netC = model[0], model[1]
    ranker.eval()

    y_pred = []
    all_probs = None

    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0

    # loop through test data stream one by one
    for i in range(len(X_stream)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        inputs = torch.from_numpy(X_stream[i]).to(torch.float32)
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).cpu()

        # accumulate test data
        if i == 0:
            data_cum = inputs.float().cpu()
        else:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)

        # Incremental EA
        if args.align:
            if i == 0:
                sample_test = data_cum.reshape(args.chn, args.time_sample_num)
            else:
                sample_test = data_cum[i].reshape(args.chn, args.time_sample_num)
            # update reference matrix
            R = EA_online(sample_test, R, i)

            sqrtRefEA = _as_real(fractional_matrix_power(R, -0.5))
            # transform current test sample
            sample_test = np.dot(sqrtRefEA, sample_test)
            sample_test = sample_test.reshape(1, 1, args.chn, args.time_sample_num)
        else:
            sample_test = data_cum[i].numpy()
            sample_test = sample_test.reshape(1, 1, sample_test.shape[1], sample_test.shape[2])

        sample_test = torch.from_numpy(sample_test).to(torch.float32).to(device)

        with torch.no_grad():
            view_logits, pred_losses = [], []
            if args.variant == 'BFT-A':
                for v in augment_branches(sample_test, args):
                    feat = netF(v)
                    _, logits = netC(feat)
                    view_logits.append(logits)
                    pred_losses.append(ranker(feat))
            else:
                feat = netF(sample_test)
                for masked in feature_mask_branches(feat, args.K):
                    _, logits = netC(masked)
                    view_logits.append(logits)
                    pred_losses.append(ranker(masked))

            pred_losses = torch.stack(pred_losses).squeeze()
            weights = F.softmax(-pred_losses, dim=0)
            all_probs = weights.unsqueeze(0) if all_probs is None \
                else torch.cat((all_probs, weights.unsqueeze(0)), 0)
            probs = all_probs.mean(dim=0)

            views = torch.stack([nn.Softmax(dim=1)(l / args.temperature) for l in view_logits]).squeeze(1)
            softmax_out = (views * probs.unsqueeze(1)).sum(dim=0) / probs.sum()
            softmax_out = softmax_out.reshape(1, args.class_num)

        y_pred.append(softmax_out.detach().cpu().numpy())

        #################### Phase 2: target model update ####################
        if (i + 1) >= args.test_batch and (i + 1) % args.stride == 0:
            if args.align:
                batch_test = np.copy(data_cum[i - args.test_batch + 1:i + 1])
                # transform test batch
                batch_test = np.dot(sqrtRefEA, batch_test)
                batch_test = np.transpose(batch_test, (1, 2, 0, 3))
            else:
                batch_test = data_cum[i - args.test_batch + 1:i + 1].numpy()
                batch_test = batch_test.reshape(args.test_batch, 1, batch_test.shape[2], batch_test.shape[3])

            batch_test = torch.from_numpy(batch_test).to(torch.float32).to(device)

            for step in range(args.steps):

                model[0].block1[2].train()
                model[0].block1[4].train()
                model[0].block2[3].train()

                with torch.no_grad():
                    model(batch_test)

                model[0].block1[2].eval()
                model[0].block1[4].eval()
                model[0].block2[3].eval()

        model.eval()

    return np.concatenate(y_pred, axis=0)


def clone_model(model):
    return copy.deepcopy(model)

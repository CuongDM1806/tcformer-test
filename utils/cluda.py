"""CLUDA (Ozyurt, Feuerriegel & Zhang, ICLR 2023) — authors' implementation.

Ported from https://github.com/oezyurty/CLUDA without changes to the method:
main/models/cluda.py (CLUDA_NN), utils/tcn_no_norm.py (TemporalConvNet),
utils/mlp.py (MLP), utils/nearest_neighbor.py (NN, sim_matrix) and
utils/augmentations.py (Augmenter). Only the device handling is generalised
(the released Augmenter hard-codes .cuda()) and the optional sequence mask and
static features, which EEG trials do not have, are dropped.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# --------------------------------------------------------------------------
# utils/tcn_no_norm.py (taken by the authors from locuslab/TCN)
# --------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, stride=1, dilation_factor=2,
                 dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = dilation_factor ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers += [TemporalBlock(in_channels, num_channels[i], kernel_size, stride=stride,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------------------------------
# utils/mlp.py, utils/nearest_neighbor.py
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, use_batch_norm=False):
        super().__init__()
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None

    def forward(self, x):
        hidden = self.input_fc(x)
        if self.use_batch_norm:
            hidden = self.batch_norm(hidden)
        y_pred = self.output_fc(F.relu(hidden))
        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)
        return y_pred


def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def NN(key, queue, num_neighbors=1, return_indices=False):
    similarity = sim_matrix(key, queue)
    indices_top_neighbors = torch.topk(similarity, k=num_neighbors, dim=1)[1]
    list_top_neighbors = [queue[indices_top_neighbors[:, i], :] for i in range(num_neighbors)]
    if return_indices:
        return torch.stack(list_top_neighbors), indices_top_neighbors
    return torch.stack(list_top_neighbors)


# --------------------------------------------------------------------------
# utils/augmentations.py (sequences are N x L x C, as in the authors' code)
# --------------------------------------------------------------------------
class Augmenter:
    """History cutout, history crop, Gaussian noise and spatial dropout."""

    def __init__(self, cutout_length=4, cutout_prob=0.5, crop_min_history=0.5, crop_prob=0.5,
                 gaussian_std=0.1, dropout_prob=0.1):
        self.cutout_length = cutout_length
        self.cutout_prob = cutout_prob
        self.crop_min_history = crop_min_history
        self.crop_prob = crop_prob
        self.gaussian_std = gaussian_std
        self.dropout_prob = dropout_prob
        self.augmentations = [self.history_cutout, self.history_crop, self.gaussian_noise,
                              self.spatial_dropout]

    def __call__(self, sequence, sequence_mask):
        for f in self.augmentations:
            sequence, sequence_mask = f(sequence, sequence_mask)
        return sequence, sequence_mask

    def history_cutout(self, sequence, sequence_mask):
        n_seq, n_len, n_channel = sequence.shape
        device = sequence.device
        cutout_start_index = torch.randint(low=0, high=n_len - self.cutout_length,
                                           size=(n_seq, 1)).expand(-1, n_len)
        cutout_end_index = cutout_start_index + self.cutout_length
        indices_tensor = torch.arange(n_len).repeat(n_seq, 1)
        mask_cutout = (indices_tensor < cutout_start_index) + (indices_tensor >= cutout_end_index)
        mask_cutout = mask_cutout.unsqueeze(dim=-1).expand(-1, -1, n_channel).long().to(device)
        cutout_selection = (torch.rand(n_seq) < self.cutout_prob).long().reshape(-1, 1, 1).to(device)
        sequence_cutout = sequence * (1 - cutout_selection) + sequence * cutout_selection * mask_cutout
        sequence_mask_cutout = (sequence_mask * (1 - cutout_selection)
                                + sequence_mask * cutout_selection * mask_cutout)
        return sequence_cutout, sequence_mask_cutout

    def history_crop(self, sequence, sequence_mask):
        n_seq, n_len, n_channel = sequence.shape
        device = sequence.device
        nonpadded = sequence_mask.sum(dim=-1).cpu()
        first_nonpadded = self.get_first_nonzero(nonpadded).reshape(-1, 1) / n_len
        crop_start_index = torch.rand(size=(n_seq, 1))
        crop_start_index = (crop_start_index * (1 - first_nonpadded) * self.crop_min_history
                            + first_nonpadded)
        crop_start_index = (crop_start_index * n_len).long().expand(-1, n_len)
        indices_tensor = torch.arange(n_len).repeat(n_seq, 1)
        mask_crop = indices_tensor >= crop_start_index
        mask_crop = mask_crop.unsqueeze(dim=-1).expand(-1, -1, n_channel).long().to(device)
        crop_selection = (torch.rand(n_seq) < self.crop_prob).long().reshape(-1, 1, 1).to(device)
        sequence_crop = sequence * (1 - crop_selection) + sequence * crop_selection * mask_crop
        sequence_mask_crop = (sequence_mask * (1 - crop_selection)
                              + sequence_mask * crop_selection * mask_crop)
        return sequence_crop, sequence_mask_crop

    def gaussian_noise(self, sequence, sequence_mask):
        padding_mask = (sequence_mask != 0).long()
        noise = nn.init.trunc_normal_(torch.empty_like(sequence), std=self.gaussian_std,
                                      a=-2 * self.gaussian_std, b=2 * self.gaussian_std)
        return sequence + padding_mask * noise, sequence_mask

    def spatial_dropout(self, sequence, sequence_mask):
        n_seq, n_len, n_channel = sequence.shape
        dropout_selection = (torch.rand((n_seq, 1, n_channel)) > self.dropout_prob).long()
        dropout_selection = dropout_selection.expand(-1, n_len, -1).to(sequence.device)
        return sequence * dropout_selection, sequence_mask * dropout_selection

    @staticmethod
    def get_first_nonzero(tensor2d):
        nonzero = tensor2d != 0
        cumsum = nonzero.cumsum(dim=-1)
        return ((cumsum == 1) & nonzero).max(dim=-1).indices


def get_augmenter(seq_len, num_channels):
    """main/algorithms.py :: CLUDA.get_augmenter (multichannel branch)."""
    cutout_len = math.floor(seq_len / 12)
    if num_channels != 1:
        return Augmenter(cutout_length=cutout_len)
    if seq_len > 1000:
        augmenter = Augmenter(cutout_length=cutout_len, cutout_prob=1, crop_min_history=0.25,
                              crop_prob=1, dropout_prob=0.0)
        augmenter.augmentations = [augmenter.history_cutout, augmenter.history_cutout,
                                   augmenter.history_cutout, augmenter.history_crop,
                                   augmenter.gaussian_noise, augmenter.spatial_dropout]
        return augmenter
    return Augmenter(cutout_length=cutout_len, dropout_prob=0.0)


# --------------------------------------------------------------------------
# main/models/cluda.py
# --------------------------------------------------------------------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class CLUDA_NN(nn.Module):

    def __init__(self, num_inputs, output_dim, num_channels, mlp_hidden_dim=256,
                 use_batch_norm=True, num_neighbors=1, kernel_size=2, stride=1,
                 dilation_factor=2, dropout=0.2, K=24576, m=0.999, T=0.07):
        super().__init__()
        self.K, self.m, self.T = K, m, T
        self.num_neighbors = num_neighbors

        self.encoder_q = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         dilation_factor=dilation_factor, dropout=dropout)
        self.encoder_k = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         dilation_factor=dilation_factor, dropout=dropout)
        self.projector = MLP(input_dim=num_channels[-1], hidden_dim=mlp_hidden_dim,
                             output_dim=num_channels[-1], use_batch_norm=use_batch_norm)
        self.predictor = MLP(input_dim=num_channels[-1], hidden_dim=mlp_hidden_dim,
                             output_dim=output_dim, use_batch_norm=use_batch_norm)
        self.discriminator = MLP(input_dim=num_channels[-1], hidden_dim=mlp_hidden_dim,
                                 output_dim=1, use_batch_norm=use_batch_norm)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue_s", F.normalize(torch.randn(num_channels[-1], K), dim=0))
        self.register_buffer("queue_t", F.normalize(torch.randn(num_channels[-1], K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        if self.training:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_t):
        if self.training:
            batch_size = keys_s.shape[0]
            ptr = int(self.queue_ptr)
            self.queue_s[:, ptr:ptr + batch_size] = keys_s.T
            self.queue_t[:, ptr:ptr + batch_size] = keys_t.T
            self.queue_ptr[0] = (ptr + batch_size) % self.K

    def encode(self, encoder, sequence):
        # N x L x C -> TCN (N x C x L) -> last time step
        return F.normalize(encoder(sequence.transpose(1, 2))[:, :, -1], dim=1)

    def forward(self, sequence_q_s, sequence_k_s, sequence_q_t, sequence_k_t, alpha):
        q_s = self.encode(self.encoder_q, sequence_q_s)
        p_q_s = F.normalize(self.projector(q_s), dim=1)
        q_t = self.encode(self.encoder_q, sequence_q_t)
        p_q_t = F.normalize(self.projector(q_t), dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_s = self.encode(self.encoder_k, sequence_k_s)
            k_t = self.encode(self.encoder_k, sequence_k_t)

        # Source / target MoCo contrastive losses.
        logits_s = torch.cat([torch.mm(p_q_s, k_s.transpose(0, 1)),
                              torch.mm(p_q_s, self.queue_s.clone().detach())], dim=1) / self.T
        labels_s = torch.arange(p_q_s.shape[0], dtype=torch.long, device=p_q_s.device)
        logits_t = torch.cat([torch.mm(p_q_t, k_t.transpose(0, 1)),
                              torch.mm(p_q_t, self.queue_t.clone().detach())], dim=1) / self.T
        labels_t = torch.arange(p_q_t.shape[0], dtype=torch.long, device=p_q_t.device)

        # Target-source nearest-neighbour contrastive loss.
        _, indices_nn = NN(k_t, q_s.clone().detach(), num_neighbors=self.num_neighbors,
                           return_indices=True)
        logits_ts = torch.mm(q_t, q_s.transpose(0, 1).clone().detach()) / self.T
        labels_ts = indices_nn.squeeze(1).to(q_t.device)

        # Adversarial domain discrimination.
        labels_domain = torch.cat([torch.ones((len(q_s), 1), device=q_s.device),
                                   torch.zeros((len(q_t), 1), device=q_t.device)], dim=0)
        q_reversed = torch.cat([ReverseLayerF.apply(q_s, alpha),
                                ReverseLayerF.apply(q_t, alpha)], dim=0)
        pred_domain = self.discriminator(q_reversed)

        y_s = self.predictor(q_s)
        self._dequeue_and_enqueue(k_s, k_t)
        return (logits_s, labels_s, logits_t, labels_t, logits_ts, labels_ts,
                pred_domain, labels_domain, y_s)

    def predict(self, sequence):
        return self.predictor(self.encode(self.encoder_q, sequence))

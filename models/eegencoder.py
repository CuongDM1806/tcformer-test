"""EEGEncoder (Liao et al., Scientific Reports 2025), unmodified architecture.

The classes below are copied verbatim from the authors' main.py
(https://github.com/BlackCattt9/EEGEncoder, commit ad7aaf4); only the training
script and data loading were left out. ``EEGEncoderBaseline`` wraps the network
for this repository's LOSO pipeline and reproduces the authors' training loss:
cross-entropy with label smoothing 0.2 on the network output plus the explicit
L2 penalties of the ``*L2`` layers, scaled by two, optimized with Adam
(lr 1e-3, batch 64).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from transformers import LlamaConfig

from .classification_module import ClassificationModule
from .eegencoder_lma import LlamaForCausalLM


class LinearL2(nn.Module):
    def __init__(self, in_features, out_features, weight_decay=0.):
        super(LinearL2, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, x):
        # 在前向传播中，我们不做任何关于权重衰减的事情
        return self.linear(x)

    def l2_loss(self):
        # 计算 L2 损失，这将在训练循环中使用
        return self.weight_decay * torch.sum(self.linear.weight ** 2)


class Conv1dL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, weight_decay=0., bias=False):
        super(Conv1dL2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=bias,
                               groups=groups)
        self.weight_decay = weight_decay

    def forward(self, x):
        # 在前向传播中，我们不做任何关于权重衰减的事情
        return self.conv1(x)

    def l2_loss(self):
        # 计算 L2 损失，这将在训练循环中使用
        return self.weight_decay * torch.sum(self.conv1.weight ** 2)


class Conv2dL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, weight_decay=0., bias=False):
        super(Conv2dL2, self).__init__()
        self.conv2 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=bias,
                               groups=groups,
                               )
        self.weight_decay = weight_decay

    def forward(self, x):
        # 在前向传播中，我们不做任何关于权重衰减的事情
        return self.conv2(x)

    def l2_loss(self):
        # 计算 L2 损失，这将在训练循环中使用
        return self.weight_decay * torch.sum(self.conv2.weight ** 2)





class ConvBlock(nn.Module):
    def __init__(self, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.5):
        super(ConvBlock, self).__init__()
        F2 = F1 * D
        # self.conv1 = nn.Conv2d(1, F1, (kernLength, 1), padding='same', bias=False)
        self.conv1 = Conv2dL2(1, F1, (kernLength, 1), padding='same', bias=False, weight_decay=0.009)

        self.batchnorm1 = nn.BatchNorm2d(F1)
        # self.depthwise = nn.Conv2d(F1, F1 * D, (1, in_chans), groups=F1, bias=False)
        self.depthwise = Conv2dL2(F1, F1 * D, (1, in_chans), groups=F1, bias=False, weight_decay=0.009)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)
        # self.conv2 = nn.Conv2d(F1 * D, F2, (16, 1), padding='same', bias=False)
        self.conv2 = Conv2dL2(F1 * D, F2, (16, 1), padding='same', bias=False, weight_decay=0.009)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        # self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.3)
        self.dp = nn.Dropout(0.3)

        model_cofig1 = LlamaConfig()
        model_cofig1.hidden_size = embed_dim
        model_cofig1.pad_token_id = 0
        model_cofig1.intermediate_size = embed_dim * 1
        model_cofig1.num_hidden_layers = 2
        model_cofig1.num_attention_heads = num_heads
        model_cofig1.vocab_size = 21
        model_cofig1.max_position_embeddings = 500
        model_cofig1.type_vocab_size = 20
        model_cofig1.dropout_ratio = 0.3
        model_cofig1.weight_decay = 0.5
        # model_cofig1.initializer_range = 0.1
        self.short_encoder = LlamaForCausalLM(config=model_cofig1)

    def forward(self, x):
        x0 = x
        # MultiheadAttention in PyTorch expects inputs of shape (L, N, E)
        # where L is the sequence length, N is the batch size, and E is the embedding dimension.
        # out, _ = self.mha(x, x, x)

        out = self.short_encoder(inputs_embeds=x, output_hidden_states=True).hidden_states[-1]
        out = self.dp(x0 + out)
        return out  # Permute back to (N, L, E)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, residual=False, apply_to_input=True):
        super(AttentionBlock, self).__init__()
        self.residual = residual
        self.apply_to_input = apply_to_input
        self.attention = MultiHeadAttentionBlock(embed_dim, num_heads)


    def forward(self, x):
        out = self.attention(x)

        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock_(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, weight_decay=0.009, max_norm=0.6, activation='relu'):
        super(TCNBlock_, self).__init__()
        self.depth = depth
        self.activation = getattr(F, activation)
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dimension, filters, 1) if input_dimension != filters else None
        self.cn1 = nn.Sequential(Conv1dL2(input_dimension, filters, kernel_size, weight_decay=0.009), nn.BatchNorm1d(filters), nn.SiLU(),nn.Dropout(0.3))
        self.cn2 = nn.Sequential(Conv1dL2(filters, filters, kernel_size, weight_decay=0.009), nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))



        for i in range(depth-1):
            dilation_size = 2 ** (i+1)
            padding = (kernel_size - 1) * dilation_size
            block_layers = [
                Conv1dL2(filters if i > 0 else input_dimension, filters, kernel_size, stride=1, padding=padding, dilation=dilation_size,
                                     weight_decay=0.009),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                Conv1dL2(filters, filters, kernel_size, stride=1, padding=padding, dilation=dilation_size, weight_decay=0.009),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            self.blocks.append(nn.Sequential(*block_layers))

        self.init_weights(max_norm)

    def init_weights(self, max_norm):
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    layer.weight.data = nn.init.kaiming_uniform_(layer.weight.data)
                    nn.utils.clip_grad_norm_(layer.parameters(), max_norm)

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.cn1(out)
        out = self.cn2(out)
        res = self.downsample(out) if self.downsample is not None else out

        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block(out)
                out += res
            else:
                out = block(out)
                out += self.blocks[i-1](res)
            out = self.activation(out)

        return out.transpose(1, 2)

class EEGEncoder(nn.Module):
    def __init__(self, n_classes=4, in_chans=22, in_samples=1125, n_windows=5, attention='mha',
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 tcn_activation='elu', fuse='average', transformer_layers=2,
                 transformer_pooling='mean', return_logits=False):
        super(EEGEncoder, self).__init__()
        if transformer_layers < 1:
            raise ValueError('transformer_layers must be at least 1')
        if transformer_pooling not in ('mean', 'last'):
            raise ValueError("transformer_pooling must be 'mean' or 'last'")
        self.n_windows = n_windows
        self.fuse = fuse
        self.transformer_pooling = transformer_pooling
        self.dense_weight_decay = 0.5
        self.from_logits = return_logits

        F2 = eegn_F1 * eegn_D

        # self.conv_block = ConvBlock(eegn_F1, eegn_F1, eegn_D, eegn_kernelSize, eegn_poolSize, eegn_dropout)
        self.conv_block = ConvBlock(F1=eegn_F1, kernLength=eegn_kernelSize, poolSize=7, D=2, in_chans=22, dropout=eegn_dropout)
        self.attention_block = AttentionBlock(embed_dim=F2, num_heads=4)  # Define your attention block
        self.tcn_blocks = nn.ModuleList([TCNBlock_(F2, tcn_depth, tcn_kernelSize, tcn_filters, tcn_dropout, tcn_activation) for _ in range(n_windows)])
        self.dense_layers = nn.ModuleList([LinearL2(tcn_filters, n_classes, 0.5) for _ in range(n_windows)])
        self.aa_drop = nn.Dropout(0.3)
        model_cofig1 = LlamaConfig()
        model_cofig1.hidden_size = F2
        model_cofig1.pad_token_id = 0
        model_cofig1.intermediate_size = F2 * 1
        model_cofig1.num_hidden_layers = transformer_layers
        model_cofig1.num_attention_heads = 2
        model_cofig1.vocab_size = 21
        model_cofig1.max_position_embeddings = 500
        model_cofig1.type_vocab_size = 20
        model_cofig1.dropout_ratio = 0.3
        model_cofig1.weight_decay = 0.5
        # model_cofig1.initializer_range = 0.1
        self.trm_block = nn.ModuleList([LlamaForCausalLM(config=model_cofig1) for _ in range(n_windows)])


        if fuse == 'concat':
            self.final_dense = LinearL2(n_classes * n_windows, n_classes, 0.5)

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1)  # Equivalent to Lambda(lambda x: x[:,:,-1,:])(block1)
        sw_outputs = []
        for i in range(self.n_windows):
            # st = i
            # end = x.shape[1] - self.n_windows + i + 1
            window_slice = self.aa_drop(x[:, :, :])
            # Apply attention if defined
            # window_slice = self.attention_block(window_slice)
            # Apply TCN block
            tcn_output = self.tcn_blocks[i](window_slice)
            tcn_output = tcn_output[:, -1, :]  # Equivalent to Lambda(lambda x: x[:,-1,:])(block3)

            trm_hidden = self.trm_block[i](
                inputs_embeds=window_slice, output_hidden_states=True
            ).hidden_states[-1]
            if self.transformer_pooling == 'last':
                trm_output = trm_hidden[:, -1, :]
            else:
                trm_output = trm_hidden.mean(1)
            tcn_output = tcn_output+F.dropout(trm_output, 0.3)
            # window_slice = window_slice-x.mean(1, keepdim=True)
            # tcn_output = self.trm_block[i](inputs_embeds=window_slice, output_hidden_states=True).hidden_states[-1]
            # tcn_output = F.relu(F.dropout(tcn_output.mean(1), 0.3))
            # tcn_output = window_slice.mean(1)
            # Apply dense layer
            dense_output = self.dense_layers[i](tcn_output)
            sw_outputs.append(dense_output)

        if self.fuse == 'average':
            out = torch.mean(torch.stack(sw_outputs, dim=0), dim=0)
        elif self.fuse == 'concat':
            out = torch.cat(sw_outputs, dim=1)
            out = self.final_dense(out)

        if not self.from_logits:
            out = F.softmax(out, dim=1)

        return out


class EEGEncoderBaseline(ClassificationModule):
    """Authors' EEGEncoder inside the shared LOSO training/evaluation pipeline."""

    def __init__(self, n_channels: int, n_classes: int, label_smoothing: float = 0.2,
                 **kwargs):
        if n_channels != 22:
            # The released ConvBlock hard-codes in_chans=22 (BCI IV-2a).
            raise ValueError("The released EEGEncoder supports 22-channel BCI IV-2a only.")
        model = EEGEncoder(n_classes=n_classes, in_chans=n_channels)
        super().__init__(model, n_classes, **kwargs)
        self.label_smoothing = label_smoothing

    def forward(self, x):
        # Pipeline tensors are [B, C, T]; the released network expects [B, 1, C, T].
        return self.model(x.unsqueeze(1))

    def shared_step(self, batch, batch_idx, mode: str = "train"):
        if mode != "train":
            return super().shared_step(batch, batch_idx, mode)
        x, y = batch
        outs = self.forward(x)
        # As in the released main.py: CrossEntropyLoss(label_smoothing=0.2)
        # applied to the network output, plus the explicit L2 terms, times 2.
        task_loss = F.cross_entropy(outs, y, label_smoothing=self.label_smoothing)
        l2_loss = sum(
            module.l2_loss() for module in self.model.modules()
            if hasattr(module, "l2_loss")
        )
        loss = 2 * (task_loss + l2_loss)
        acc = accuracy(outs, y, task="multiclass", num_classes=self.hparams.n_classes)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss, acc

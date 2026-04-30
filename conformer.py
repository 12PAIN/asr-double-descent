import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_length(lengths, kernel_size=3, stride=2, padding=1, dilation=1):
    return torch.floor((lengths + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1).to(
        torch.long
    )


class Swish(nn.Module):
    def forward(self, x):
        return F.silu(x)


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class ConvSubsampling4(nn.Module):
    def __init__(self, in_feats, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # F: in_feats -> F1=floor((F+1)/2) -> F2=floor((F1+1)/2)
        F1 = (in_feats + 1) // 2
        F2 = (F1 + 1) // 2
        self._proj = nn.Linear(d_model * F2, d_model)

    def forward(self, x, lengths):
        B, T, Freq = x.shape
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.conv(x)  # [B, D, T', F']
        B, D, Tp, Fp = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T', D, F']
        x = x.view(B, Tp, D * Fp)  # [B, T', D*F']
        x = self._proj(x)  # [B, T', D]

        lengths = conv_out_length(lengths, kernel_size=3, stride=2, padding=1)
        lengths = conv_out_length(lengths, kernel_size=3, stride=2, padding=1)
        return x, lengths


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """
    Conformer convolution module:
    pointwise conv -> GLU -> depthwise conv -> BN -> Swish -> pointwise conv -> dropout
    """

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding"
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size) // 2,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, D] -> conv1d expects [B, D, T]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise1(x)  # [B, 2D, T]
        x = F.glu(x, dim=1)  # [B, D, T]
        if key_padding_mask is not None:
            # zero before depthwise: prevents padded frames from contaminating
            # valid frames at sequence boundaries through the conv kernel
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        x = self.depthwise(x)  # [B, D, T]
        if key_padding_mask is not None:
            # zero again before BN: depthwise output at padded positions is non-zero
            # (conv kernel partially overlaps valid region); BN running stats must
            # reflect only real frames
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pointwise2(x)
        x = self.drop(x)
        return x.transpose(1, 2)  # back to [B, T, D]


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, D]
        # key_padding_mask: [B, T], True where padding
        y, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.drop(y)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_expansion=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, expansion_factor=ff_expansion, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, expansion_factor=ff_expansion, dropout=dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout=dropout)
        self.conv = ConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)

        self.norm_ff1 = nn.LayerNorm(d_model)
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Pre-norm + residual, FF scaled by 0.5 as in Conformer
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x = x + self.mhsa(self.norm_mhsa(x), key_padding_mask=key_padding_mask)
        x = x + self.conv(self.norm_conv(x), key_padding_mask=key_padding_mask)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        x = self.norm_out(x)
        return x


class ConformerCTC(nn.Module):
    def __init__(
        self,
        in_feats,
        vocab_size,
        d_model=256,
        n_heads=4,
        num_layers=6,
        dropout=0.1,
        conv_kernel=31,
    ):
        super().__init__()
        self.subsample = ConvSubsampling4(in_feats, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    n_heads,
                    ff_expansion=4,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, features, feature_lengths):
        # features: [B, T, F]
        x, out_lengths = self.subsample(features, feature_lengths)  # [B, T', D]
        x = self.posenc(x)
        x = self.drop(x)

        # key_padding_mask: True там, где padding
        B, Tp, _ = x.shape
        key_padding_mask = torch.arange(Tp, device=x.device).unsqueeze(0).expand(
            B, Tp
        ) >= out_lengths.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        logits = self.ctc_head(x)  # [B, T', V]
        return logits, out_lengths


def ctc_greedy_decode(logits, out_lengths, blank_id=0):
    # logits: [B, T, V]
    pred = torch.argmax(logits, dim=-1)  # [B, T]
    hyps = []
    for b in range(pred.size(0)):
        T = out_lengths[b].item()
        seq = pred[b, :T].tolist()
        collapsed = []
        prev = None
        for t in seq:
            if t == prev:
                continue
            if t != blank_id:
                collapsed.append(t)
            prev = t
        hyps.append(collapsed)
    return hyps


def ctc_beam_search_decode(logits, out_lengths, blank_id=0, beam_size=10):
    """CTC prefix beam search (Graves 2012). No LM.

    Each beam entry tracks (p_b, p_nb): probability mass of all CTC paths
    producing that prefix ending in blank / non-blank.

    Speed: only the top-k tokens per frame are expanded (k = min(4*beam+4, V)).
    This cuts the inner loop ~25x versus iterating all V tokens, with negligible
    accuracy loss (skipped tokens have near-zero probability).
    """
    import numpy as np

    probs = logits.float().softmax(dim=-1)  # [B, T, V]
    V = probs.size(-1)
    topk = min(beam_size * 4 + 4, V)
    results = []

    for b in range(probs.size(0)):
        T = out_lengths[b].item()
        p = probs[b, :T].cpu().numpy()  # [T, V] float32

        beam = {(): (1.0, 0.0)}  # prefix_tuple -> (p_b, p_nb)

        for t in range(T):
            pt = p[t]  # [V] numpy array

            # top-k indices for this frame; blank is always included separately
            top_idx = np.argpartition(pt, -topk)[-topk:].tolist()

            next_beam = {}
            pruned = sorted(beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_size]

            for prefix, (pb, pnb) in pruned:
                p_total = pb + pnb
                last = prefix[-1] if prefix else None

                # blank: prefix unchanged, contributes to p_b
                new_pb = p_total * float(pt[blank_id])
                cur = next_beam.get(prefix, (0.0, 0.0))
                next_beam[prefix] = (cur[0] + new_pb, cur[1])

                # non-blank tokens (top-k only)
                for c in top_idx:
                    if c == blank_id:
                        continue
                    pc = float(pt[c])
                    if c == last:
                        # same token without blank: stays at prefix, contributes to p_nb
                        cur = next_beam.get(prefix, (0.0, 0.0))
                        next_beam[prefix] = (cur[0], cur[1] + pnb * pc)
                        # blank-separated repeat: extends prefix
                        new_prefix = prefix + (c,)
                        cur2 = next_beam.get(new_prefix, (0.0, 0.0))
                        next_beam[new_prefix] = (cur2[0], cur2[1] + pb * pc)
                    else:
                        new_prefix = prefix + (c,)
                        cur = next_beam.get(new_prefix, (0.0, 0.0))
                        next_beam[new_prefix] = (cur[0], cur[1] + p_total * pc)

            beam = next_beam

        best = max(beam.items(), key=lambda x: x[1][0] + x[1][1])[0]
        results.append(list(best))

    return results

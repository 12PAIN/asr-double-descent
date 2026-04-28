import math
import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm
from conformer import ctc_greedy_decode


# ---------------------------------------------------------------------------
# Architecture complexity and Rademacher bound (Section 4-5 of the paper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _spectral_norm_matrix(W: torch.Tensor) -> float:
    """Largest singular value of a weight tensor, reshaped to 2D."""
    if W.dim() == 1:
        return float(W.abs().max())
    W2 = W.float().reshape(W.size(0), -1)
    return float(torch.linalg.svdvals(W2).max())


@torch.no_grad()
def compute_conformer_complexity(model, n_train: int, T_max: int, K: int) -> dict:
    """
    Compute C_Conf and C_G for the Rademacher bound (Corollary 2 in the paper).

    C_Conf = (prod_l Lambda_l) * (sum_l Delta_l^{2/3})^{3/2}
    C_G = C_Conf^2 * log(n * d * T_max * K)

    Per-block Lipschitz constants use spectral norms of weight matrices:
      - FF: ||W2||_s * Swish_Lip * ||W1||_s   (Swish_Lip <= 1.1)
      - MHSA: ||W_in||_s * ||W_out||_s
      - Conv: ||W_pw1||_s * max_c(||w_c||_1) * ||BN||_Lip * Swish_Lip * ||W_pw2||_s
        where max_c(||w_c||_1) is the Toeplitz spectral bound (||w||_1 for each channel).
    Residual connections: L_r^res = 1 + scale * L_r (FF modules use scale=0.5).
    LayerNorm is 1-Lipschitz.

    NOTE: The bound is conditional on Assumption 2 (covering lemma for the Toeplitz
    convolution submodule -- the main open step acknowledged in the paper).
    In practice C_G is large, so the bound may be vacuous for big models; it is
    an architectural upper bound, not a tight estimate.
    """
    d = model.ctc_head.in_features
    swish_lip = 1.1  # empirical upper bound on max |d/dx [x * sigmoid(x)]|

    Lambda_list = []
    Delta_list = []

    for block in model.layers:
        # --- per-submodule Lipschitz constants (inner, before scaling by residual) ---

        # FF1: Linear(d,4d) -> Swish -> Linear(4d,d)
        L_ff1 = (
            _spectral_norm_matrix(block.ff1.net[0].weight)
            * swish_lip
            * _spectral_norm_matrix(block.ff1.net[3].weight)
        )

        # MHSA: in_proj [3d,d] and out_proj [d,d]
        mha = block.mhsa.mha
        L_mhsa = (
            _spectral_norm_matrix(mha.in_proj_weight)
            * _spectral_norm_matrix(mha.out_proj.weight)
        )

        # Conv: pw1 -> GLU -> depthwise -> BN -> Swish -> pw2
        W_pw1 = block.conv.pointwise1.weight.squeeze(-1)  # [2d, d]
        L_pw1 = _spectral_norm_matrix(W_pw1)

        # Depthwise: weight [d, 1, k]; Toeplitz spectral bound = max_c ||w_c||_1
        w_dw = block.conv.depthwise.weight.squeeze(1)  # [d, k]
        L_dw = float(w_dw.abs().sum(dim=1).max())

        # BatchNorm: |gamma_c| / sqrt(running_var_c + eps) per channel
        bn = block.conv.bn
        if bn.running_var is not None and bn.weight is not None:
            L_bn = float((bn.weight.abs() / (bn.running_var + bn.eps).sqrt()).max())
        else:
            L_bn = 1.0

        W_pw2 = block.conv.pointwise2.weight.squeeze(-1)  # [d, d]
        L_pw2 = _spectral_norm_matrix(W_pw2)
        L_conv = L_pw1 * L_dw * L_bn * swish_lip * L_pw2

        # FF2: same structure as FF1
        L_ff2 = (
            _spectral_norm_matrix(block.ff2.net[0].weight)
            * swish_lip
            * _spectral_norm_matrix(block.ff2.net[3].weight)
        )

        # --- residual Lipschitz constants: L_r^res = 1 + scale * L_r ---
        # Order: ff1 (scale=0.5), mhsa (scale=1), conv (scale=1), ff2 (scale=0.5), norm_out (1-Lip)
        L_res = [
            1.0 + 0.5 * L_ff1,   # FF1 residual
            1.0 + L_mhsa,         # MHSA residual
            1.0 + L_conv,         # Conv residual
            1.0 + 0.5 * L_ff2,   # FF2 residual
            1.0,                  # output LayerNorm: 1-Lipschitz, no bypass
        ]
        # "inner" Lipschitz (the part that enters the deviation factor a_r)
        L_inner = [0.5 * L_ff1, L_mhsa, L_conv, 0.5 * L_ff2, 0.0]

        # Lambda_l = prod_r L_r^res
        Lambda_l = 1.0
        for lr in L_res:
            Lambda_l *= lr
        Lambda_list.append(Lambda_l)

        # C_blk_l = (sum_r (alpha_r * L_r_inner)^{2/3})^{3/2}
        # alpha_r = prod_{s > r} L_s^res  (tail product of residual constants)
        terms = []
        for r in range(len(L_res)):
            alpha_r = 1.0
            for s in range(r + 1, len(L_res)):
                alpha_r *= L_res[s]
            terms.append((alpha_r * L_inner[r]) ** (2.0 / 3.0))
        Delta_list.append(sum(terms) ** (3.0 / 2.0))

    # C_Conf = (prod_l Lambda_l) * (sum_l Delta_l^{2/3})^{3/2}
    prod_Lambda = 1.0
    for lam in Lambda_list:
        prod_Lambda *= lam

    sum_Delta_23 = sum(dl ** (2.0 / 3.0) for dl in Delta_list)
    C_conf = prod_Lambda * (sum_Delta_23 ** (3.0 / 2.0))

    log_factor = math.log(max(n_train * d * T_max * K, 2))
    C_G = C_conf ** 2 * log_factor

    return {
        "C_conf": C_conf,
        "C_G": C_G,
        "Lambda_per_block": Lambda_list,
        "Delta_per_block": Delta_list,
        "log_factor": log_factor,
    }


def rademacher_gen_bound(
    C_G: float,
    n: int,
    B: float,
    delta: float = 0.05,
    C_prime: float = 4.0,
) -> dict:
    """
    Rademacher generalization bound, Eq. (1) in the paper:

      G(theta, S) <= C' * sqrt(C_G/n) * (1 + log(B/sqrt(C_G/n)))
                   + B * sqrt(log(1/delta) / n)

    This follows from Dudley's entropy integral with optimal alpha* = sqrt(C_G/n).
    The paper writes the log argument as B*sqrt(n/C_G) = B/alpha*, which is equivalent.

    When alpha* >= B (i.e. C_G/n >= B^2), the Dudley infimum is achieved at alpha=B,
    giving the trivial diameter bound: Rademacher complexity <= B. In that case the
    Rademacher term is vacuous (C'*B). For large networks this is always the case.

    C_prime: constant from the bounded-loss Rademacher inequality + Dudley integral;
             the paper writes C' without specifying its value (typically 2-8).
    delta: failure probability.

    Returns a dict with the full bound and each additive term.
    """
    alpha_star = math.sqrt(max(C_G / n, 1e-300))  # optimal Dudley threshold
    if alpha_star >= B:
        # Dudley integral collapses: infimum at alpha=B gives trivial bound C'*B
        rademacher_term = C_prime * B
        vacuous = True
    else:
        # alpha_star < B: log(B/alpha_star) > 0, formula is valid
        rademacher_term = C_prime * alpha_star * (1.0 + math.log(B / alpha_star))
        vacuous = False
    confidence_term = B * math.sqrt(math.log(max(1.0 / delta, 1.0 + 1e-10)) / n)
    return {
        "gen_bound": rademacher_term + confidence_term,
        "rademacher_term": rademacher_term,
        "confidence_term": confidence_term,
        "sqrt_CG_over_n": alpha_star,
        "vacuous": vacuous,
    }


class WarmupCosineScheduler:
    """Linear warmup (pct_start of steps) + cosine decay to min_lr. Step-based."""

    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        max_lr,
        min_lr=1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _get_lr(self):
        if self._step <= self.warmup_steps:
            return self.max_lr * (self._step / max(1, self.warmup_steps))
        progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * min(1.0, progress))
        )

    def get_last_lr(self):
        return [self._get_lr()]


def compute_wer(
    predictions: list[str],
    references: list[str],
    wer_metric=None,
) -> float:
    """Compute Word Error Rate using HuggingFace evaluate (jiwer)."""
    if wer_metric is None:
        wer_metric = evaluate.load("wer")
    out = wer_metric.compute(predictions=predictions, references=references)
    return float(out["wer"]) if isinstance(out, dict) else float(out)


@torch.no_grad()
def evaluate_dataloader(
    model,
    dl,
    sp,
    device,
    blank_id=0,
    wer_metric=None,
    max_batches=None,
    use_amp=True,
    desc="Eval",
    return_texts=False,
    use_keep_mask=False,
    clip_B=None,
):
    """
    Single pass over a dataloader: forward, CTC loss, and optionally greedy decode + WER.

    Also computes norm_loss = mean(ell_CTC / T) and clipped_norm_loss = mean(min(ell_CTC/T, B)),
    where T is the logit sequence length (after subsampling). These match the paper's clipped
    normalized CTC loss ell_B used in the three-term decomposition.

    With return_texts=False (default) the decode and sp.decode calls are skipped entirely;
    "wer" is NaN and "predictions"/"references" are empty lists.
    With return_texts=True behaviour is unchanged: full decode, WER, and text lists returned.
    With use_keep_mask=False the keep = out_lengths >= target_lengths guard is skipped
    entirely (no GPU -> CPU sync); CTCLoss(zero_infinity=True) handles short sequences.
    """
    model.eval()
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True).to(device)
    # reduction="none" gives raw per-sample -log p(z|x) without any normalization
    ctc_none = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True).to(device)
    total_loss = 0.0
    total_norm_loss = 0.0
    total_clipped_norm_loss = 0.0
    n_loss = 0
    predictions = []
    references = []

    for batch_idx, (
        features,
        feature_lengths,
        targets,
        target_lengths,
    ) in enumerate(tqdm(dl, total=len(dl), desc=desc)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        features = features.to(device, non_blocking=True)
        feature_lengths = feature_lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, out_lengths = model(features, feature_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T', B, K]
            if use_keep_mask:
                keep = out_lengths >= target_lengths
                if keep.sum().item() > 0:
                    loss = ctc(
                        log_probs[:, keep, :],
                        targets[keep],
                        out_lengths[keep],
                        target_lengths[keep],
                    )
                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        # per-sample normalized loss for kept samples
                        per_s = ctc_none(
                            log_probs[:, keep, :], targets[keep],
                            out_lengths[keep], target_lengths[keep],
                        ).float()
                        T_kept = out_lengths[keep].float().clamp_min(1)
                        norm = per_s / T_kept  # ell_bar_CTC = ell_CTC / T
                        clipped = norm.clamp_max(clip_B) if clip_B is not None else norm
                        finite = norm.isfinite()
                        if finite.any():
                            total_norm_loss += norm[finite].mean().item()
                            total_clipped_norm_loss += clipped[finite].mean().item()
                        n_loss += 1
            else:
                loss = ctc(log_probs, targets, out_lengths, target_lengths)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    # per-sample normalized loss: ell_bar_CTC = ell_CTC / T
                    per_s = ctc_none(log_probs, targets, out_lengths, target_lengths).float()
                    T_all = out_lengths.float().clamp_min(1)
                    norm = per_s / T_all
                    clipped = norm.clamp_max(clip_B) if clip_B is not None else norm
                    finite = norm.isfinite()
                    if finite.any():
                        total_norm_loss += norm[finite].mean().item()
                        total_clipped_norm_loss += clipped[finite].mean().item()
                    n_loss += 1

        if return_texts:
            hyps = ctc_greedy_decode(logits, out_lengths, blank_id=blank_id)
            tgt_lens = target_lengths.tolist()  # one D2H transfer
            targets_cpu = targets.cpu()  # one D2H transfer
            if use_keep_mask:
                keep_cpu = keep.tolist()
                valid_hyps = [hyps[i] for i in range(len(hyps)) if keep_cpu[i]]
                valid_tgts = [
                    targets_cpu[i, : tgt_lens[i]].tolist() for i in range(len(hyps)) if keep_cpu[i]
                ]
            else:
                valid_hyps = hyps
                valid_tgts = [targets_cpu[i, : tgt_lens[i]].tolist() for i in range(len(hyps))]
            if valid_hyps:
                predictions.extend(sp.decode(valid_hyps))  # one batched call
                references.extend(sp.decode(valid_tgts))  # one batched call

    mean_loss = total_loss / max(n_loss, 1)
    wer = (
        compute_wer(predictions, references, wer_metric=wer_metric) if predictions else float("nan")
    )
    return {
        "loss": mean_loss,
        "norm_loss": total_norm_loss / max(n_loss, 1),
        "clipped_norm_loss": total_clipped_norm_loss / max(n_loss, 1),
        "wer": wer,
        "predictions": predictions,
        "references": references,
    }


@torch.no_grad()
def compute_wer_and_transcripts(
    model, dl, sp, device, blank_id=0, max_batches=None, wer_metric=None
):
    """
    Run greedy decode over a dataloader; return WER and lists of prediction/reference strings.
    Prefer evaluate_dataloader() for loss + WER in one pass.
    """
    out = evaluate_dataloader(
        model,
        dl,
        sp,
        device,
        blank_id=blank_id,
        wer_metric=wer_metric,
        max_batches=max_batches,
        desc="WER",
        return_texts=True,
    )
    return out["wer"], out["predictions"], out["references"]


@torch.no_grad()
def demo_decode_batch(model, batch, sp, device, blank_id=0):
    model.eval()
    features, feature_lengths, targets, target_lengths = batch
    features = features.to(device, non_blocking=True)
    feature_lengths = feature_lengths.to(device, non_blocking=True)

    logits, out_lengths = model(features, feature_lengths)
    hyps = ctc_greedy_decode(logits, out_lengths, blank_id=blank_id)

    # покажем 2-3 примера
    for i in range(min(3, len(hyps))):
        hyp_text = sp.decode(hyps[i])
        # target -> соберём реальные id без pad (0)
        tgt = targets[i, : target_lengths[i]].tolist()
        ref_text = sp.decode(tgt)
        print("REF:", ref_text)
        print("HYP:", hyp_text)
        print("---")


def train_one_step(
    model,
    batch,
    optimizer,
    scaler,
    device,
    blank_id,
    use_amp,
    ctc,
    use_keep_mask=False,
):
    """One training step. Returns (loss_value, did_step, skip_reason).
    skip_reason: "ok" | "keep0" | "nan" | "overflow" for logging.
    With use_keep_mask=False the keep guard (and its GPU -> CPU sync) is skipped;
    CTCLoss(zero_infinity=True) handles any short sequences silently."""
    model.train()
    features, feature_lengths, targets, target_lengths = batch
    features = features.to(device, non_blocking=True)
    feature_lengths = feature_lengths.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    target_lengths = target_lengths.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits, out_lengths = model(features, feature_lengths)
        if use_keep_mask:
            keep = out_lengths >= target_lengths
            if keep.sum().item() == 0:
                return None, False, "keep0"
            logits = logits[keep]
            out_lengths = out_lengths[keep]
            targets = targets[keep]
            target_lengths = target_lengths[keep]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = ctc(log_probs, targets, out_lengths, target_lengths)
    if not torch.isfinite(loss):
        return None, False, "nan"

    scale_before = scaler.get_scale() if scaler.is_enabled() else None
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    did_step = (not scaler.is_enabled()) or (scaler.get_scale() >= scale_before)
    skip_reason = "overflow" if not did_step else "ok"
    return loss.item(), did_step, skip_reason


def train_one_epoch(
    model,
    dl,
    optimizer,
    scheduler,
    device,
    blank_id=0,
    use_amp=True,
    scaler=None,
):
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True).to(device)
    total_loss = 0.0
    n = 0
    for batch in tqdm(dl, total=len(dl)):
        loss_val, did_step, _ = train_one_step(
            model, batch, optimizer, scaler, device, blank_id, use_amp, ctc
        )
        if did_step:
            scheduler.step()
        if loss_val is not None:
            total_loss += loss_val
            n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def compute_loss_over_dataloader(
    model, dl, device, blank_id=0, use_amp=True, max_batches=None, use_keep_mask=False
):
    """Compute mean CTC loss over a dataloader (e.g. train or test)."""
    model.eval()
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True).to(device)
    total_loss = 0.0
    n = 0
    for batch_idx, (
        features,
        feature_lengths,
        targets,
        target_lengths,
    ) in enumerate(dl):
        if max_batches is not None and batch_idx >= max_batches:
            break
        features = features.to(device, non_blocking=True)
        feature_lengths = feature_lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, out_lengths = model(features, feature_lengths)
            if use_keep_mask:
                keep = out_lengths >= target_lengths
                if keep.sum().item() == 0:
                    continue
                logits = logits[keep]
                out_lengths = out_lengths[keep]
                targets = targets[keep]
                target_lengths = target_lengths[keep]
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = ctc(log_probs, targets, out_lengths, target_lengths)
        if torch.isfinite(loss):
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def weight_norm_sum(model):
    """Global L2 norm of all trainable parameters: sqrt(sum(||p||_2^2))."""
    return sum(p.data.norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5


def ctc_lipschitz_proxy(model, dl, device, blank_id=0, n_batches=5):
    """
    Empirical mean of ||nabla_U ell_bar_CTC(x,z;U)||_F per sample.
    Proposition 2 proves this is <= sqrt(2/T) <= sqrt(2).
    Uses CTC backprop identity: nabla_{u_t} ell_CTC = y_t - gamma_t,
    so the gradient w.r.t. the logit tensor U gives (y - gamma) directly.
    Requires one backward pass per batch; does not update model weights.
    """
    model.eval()
    ctc_none = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True).to(device)
    norms = []
    for batch_idx, (features, feature_lengths, targets, target_lengths) in enumerate(dl):
        if batch_idx >= n_batches:
            break
        features = features.to(device)
        feature_lengths = feature_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.no_grad():
            logits, out_lengths = model(features, feature_lengths)

        # Compute gradient w.r.t. the logit tensor U (not model parameters).
        # CTCLoss backprop gives: grad[b, t, k] = d ell_CTC(b) / d U[b,t,k] = y[b,t,k] - gamma[b,t,k]
        U = logits.detach().float().requires_grad_(True)  # [B, T', K]
        log_probs = U.log_softmax(dim=-1).transpose(0, 1)  # [T', B, K]
        per_sample = ctc_none(log_probs, targets, out_lengths, target_lengths)  # [B]
        per_sample.sum().backward()

        if U.grad is not None:
            with torch.no_grad():
                for b in range(U.size(0)):
                    T_b = int(out_lengths[b].item())
                    if T_b <= 0:
                        continue
                    grad_b = U.grad[b, :T_b, :]  # [T_b, K]: d ell / d U_b
                    if grad_b.isfinite().all():
                        # (1/T) * ||d ell / d U_b||_F = ||d ell_bar / d U_b||_F
                        norms.append(grad_b.norm(p="fro").item() / T_b)
    return float(sum(norms) / len(norms)) if norms else float("nan")


def sharpness_proxy(
    model,
    dl,
    device,
    blank_id=0,
    eps=1e-3,
    n_batches=10,
    n_perturbations=3,
    relative_eps=False,
    use_keep_mask=False,
):
    """
    One-sided sharpness proxy: E[ max(0, L(theta+delta) - L(theta)) ].
    - Global L2 perturbation: total ||delta|| = eps (or eps * ||theta|| if relative_eps).
    - Same keep mask for orig and perturbed loss (no recompute after perturb).
    - K perturbations per batch, then average.
    - Perturb/restore done fully on device via add/sub - no CPU state_dict copy.
    - With use_keep_mask=False the keep guard (and its GPU -> CPU sync) is skipped.
    """
    model.eval()
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # theta_norm is constant: params are fully restored after every perturbation.
    with torch.no_grad():
        theta_norm = sum(p.data.norm().item() ** 2 for p in params) ** 0.5 if relative_eps else None
    deltas = []
    batch_count = 0
    for features, feature_lengths, targets, target_lengths in dl:
        if batch_count >= n_batches:
            break
        features = features.to(device, non_blocking=True)
        feature_lengths = feature_lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        with torch.no_grad():
            logits, out_lengths = model(features, feature_lengths)
            if use_keep_mask:
                keep = out_lengths >= target_lengths
                if keep.sum().item() == 0:
                    continue
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        if use_keep_mask:
            loss_orig = ctc(
                log_probs[:, keep, :],
                targets[keep],
                out_lengths[keep],
                target_lengths[keep],
            ).item()
        else:
            loss_orig = ctc(log_probs, targets, out_lengths, target_lengths).item()

        batch_deltas = []
        for _ in range(n_perturbations):
            with torch.no_grad():
                noises = [torch.randn_like(p) for p in params]
                global_norm = sum(n.norm().item() ** 2 for n in noises) ** 0.5
                if global_norm < 1e-12:
                    continue
                scale = (eps * theta_norm) / global_norm if relative_eps else eps / global_norm
                for p, noise in zip(params, noises):
                    p.data.add_(noise, alpha=scale)
            with torch.no_grad():
                logits_p, _ = model(features, feature_lengths)
                log_probs_p = logits_p.log_softmax(dim=-1).transpose(0, 1)
                if use_keep_mask:
                    loss_pert = ctc(
                        log_probs_p[:, keep, :],
                        targets[keep],
                        out_lengths[keep],
                        target_lengths[keep],
                    ).item()
                else:
                    loss_pert = ctc(log_probs_p, targets, out_lengths, target_lengths).item()
                batch_deltas.append(max(0.0, loss_pert - loss_orig))
            # Restore in-place on device - avoids load_state_dict's CPU -> GPU round-trip.
            with torch.no_grad():
                for p, noise in zip(params, noises):
                    p.data.sub_(noise, alpha=scale)
        if batch_deltas:
            deltas.extend(batch_deltas)
        batch_count += 1
    return float(sum(deltas) / len(deltas)) if deltas else float("nan")

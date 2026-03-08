import math
import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm
from conformer import ctc_greedy_decode


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
):
    """
    Single pass over a dataloader: forward, CTC loss, and optionally greedy decode + WER.
    With return_texts=False (default) the decode and sp.decode calls are skipped entirely;
    "wer" is NaN and "predictions"/"references" are empty lists.
    With return_texts=True behaviour is unchanged: full decode, WER, and text lists returned.
    With use_keep_mask=False the keep = out_lengths >= target_lengths guard is skipped
    entirely (no GPU -> CPU sync); CTCLoss(zero_infinity=True) handles short sequences.
    """
    model.eval()
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True).to(device)
    total_loss = 0.0
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
            if use_keep_mask:
                keep = out_lengths >= target_lengths
                if keep.sum().item() > 0:
                    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                    loss = ctc(
                        log_probs[:, keep, :],
                        targets[keep],
                        out_lengths[keep],
                        target_lengths[keep],
                    )
                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        n_loss += 1
            else:
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = ctc(log_probs, targets, out_lengths, target_lengths)
                if torch.isfinite(loss):
                    total_loss += loss.item()
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
    """Sum of squared L2 norms of all parameters (as in proposal: weight norms)."""
    return sum(p.data.norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5


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

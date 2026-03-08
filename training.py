"""
Shared step-based training loop for Conformer CTC.
Used by train.py (single run, CLI) and sweep_run.py (multi run, config JSON).
"""


def _infinite_dataloader(dl):
    """Cycle a DataLoader without accumulating batches (itertools.cycle saves all
    yielded items in an internal list, causing unbounded pinned-memory growth).
    """
    while True:
        yield from dl


import torch
import evaluate
from tqdm import tqdm
import random, numpy as np
from conformer import ConformerCTC
from utils import (
    WarmupCosineScheduler,
    evaluate_dataloader,
    sharpness_proxy,
    train_one_step,
    weight_norm_sum,
)

# Defaults aligned with train.py / sweep config
DEFAULT_TRAIN_CONFIG = {
    "max_steps": 100_000,
    "eval_every": 5000,
    "scheduler": "warmup_cosine",
    "batch_size": 16,
    "eval_batch_size": None,
    "lr": 5e-5,
    "weight_decay": 1e-6,
    "max_lr": 1e-4,
    "max_eval_batches": 50,
    "sharpness_n_batches": 10,
    "sharpness_eps": 1e-3,
    "sharpness_relative_eps": True,
}

DEFAULT_MODEL_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "num_layers": 16,
    "dropout": 0.0,
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_config, vocab_size, device):
    """Build ConformerCTC from config dict."""
    cfg = {**DEFAULT_MODEL_CONFIG, **model_config}
    model = ConformerCTC(
        in_feats=80,
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    return model


def build_optimizer_and_scheduler(model, train_config, device):
    """Build AdamW optimizer and LR scheduler from train_config."""
    tc = {**DEFAULT_TRAIN_CONFIG, **train_config}
    lr = tc["lr"]
    weight_decay = tc["weight_decay"]
    max_lr = tc["max_lr"]
    max_steps = tc["max_steps"]
    scheduler_name = tc["scheduler"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if scheduler_name == "one_cycle":
        from torch.optim.lr_scheduler import OneCycleLR

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
    else:
        warmup_steps = max(1, int(0.05 * max_steps))
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            max_lr=max_lr,
            min_lr=1e-6,
        )
    return optimizer, scheduler


def run_training(
    model_config,
    train_config,
    train_dl,
    train_eval_dl,
    test_dl,
    sp,
    blank_id,
    vocab_size,
    device,
    seed=None,
    progress_desc="Steps",
    on_eval_callback=None,
    val_dl=None,
):
    """
    Run step-based training. Shared by train.py and sweep_run.py.

    Args:
        model_config: dict (d_model, n_heads, num_layers, dropout).
        train_config: dict (max_steps, eval_every, scheduler, lr, ...).
        train_dl, train_eval_dl, test_dl: DataLoaders.
        sp, blank_id, vocab_size: tokenizer and vocab.
        device: torch device.
        seed: optional; if set, torch.manual_seed(seed) before building model.
        progress_desc: tqdm description.
        on_eval_callback: optional callable(step_1based, row, model) after each
            eval (e.g. to save checkpoint).

    Returns:
        (model, step_logs) where step_logs is list of dicts per eval.
    """
    tc = {**DEFAULT_TRAIN_CONFIG, **train_config}
    max_steps = tc["max_steps"]
    eval_every = tc["eval_every"]
    max_eval_batches = tc["max_eval_batches"]
    sharpness_n_batches = tc["sharpness_n_batches"]
    sharpness_eps = tc["sharpness_eps"]
    sharpness_relative_eps = tc["sharpness_relative_eps"]
    use_keep_mask = tc.get("use_keep_mask", False)

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = build_model(model_config, vocab_size, device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, train_config, device
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    wer_metric = evaluate.load("wer")
    ctc = torch.nn.CTCLoss(
        blank=blank_id, reduction="mean", zero_infinity=True
    ).to(device)

    it = _infinite_dataloader(train_dl)
    step_logs = []
    running_loss_step = 0.0
    running_n_step = 0
    running_loss_iter = 0.0
    running_n_iter = 0
    opt_step = 0
    iters_in_interval = 0
    opt_steps_in_interval = 0
    skipped_overflow = 0
    skipped_keep0 = 0
    skipped_nan = 0

    pbar = tqdm(total=max_steps, desc=progress_desc)
    while opt_step < max_steps:
        batch = next(it)
        iters_in_interval += 1
        loss_val, did_step, skip_reason = train_one_step(
            model,
            batch,
            optimizer,
            scaler,
            device,
            blank_id,
            use_amp,
            ctc,
            use_keep_mask=use_keep_mask,
        )
        if skip_reason == "overflow":
            skipped_overflow += 1
        elif skip_reason == "keep0":
            skipped_keep0 += 1
        elif skip_reason == "nan":
            skipped_nan += 1

        if did_step and (loss_val is not None):
            running_loss_step += loss_val
            running_n_step += 1
        if loss_val is not None:
            running_loss_iter += loss_val
            running_n_iter += 1

        if did_step:
            opt_step += 1
            opt_steps_in_interval += 1
            scheduler.step()
            pbar.update(1)

            if (opt_step % eval_every == 0) or (opt_step == max_steps):
                train_loss_step_avg = (
                    running_loss_step / running_n_step
                    if running_n_step
                    else float("nan")
                )
                train_loss_iter_avg = (
                    running_loss_iter / running_n_iter
                    if running_n_iter
                    else float("nan")
                )
                iters_per_opt_step = (
                    iters_in_interval / opt_steps_in_interval
                    if opt_steps_in_interval
                    else float("nan")
                )
                running_loss_step = 0.0
                running_n_step = 0
                running_loss_iter = 0.0
                running_n_iter = 0
                iters_in_interval = 0
                opt_steps_in_interval = 0
                interval_overflow = skipped_overflow
                interval_keep0 = skipped_keep0
                interval_nan = skipped_nan
                skipped_overflow = 0
                skipped_keep0 = 0
                skipped_nan = 0

                train_metrics = evaluate_dataloader(
                    model,
                    train_eval_dl,
                    sp,
                    device,
                    blank_id=blank_id,
                    wer_metric=wer_metric,
                    max_batches=max_eval_batches,
                    use_amp=use_amp,
                    desc="Train eval",
                    return_texts=True,
                    use_keep_mask=use_keep_mask,
                )
                if val_dl is not None:
                    val_metrics = evaluate_dataloader(
                        model,
                        val_dl,
                        sp,
                        device,
                        blank_id=blank_id,
                        wer_metric=wer_metric,
                        max_batches=max_eval_batches,
                        use_amp=use_amp,
                        desc="Val eval",
                        return_texts=True,
                        use_keep_mask=use_keep_mask,
                    )
                else:
                    val_metrics = {"loss": float("nan"), "wer": float("nan")}
                test_metrics = evaluate_dataloader(
                    model,
                    test_dl,
                    sp,
                    device,
                    blank_id=blank_id,
                    wer_metric=wer_metric,
                    max_batches=max_eval_batches,
                    use_amp=use_amp,
                    desc="Test eval",
                    return_texts=True,
                    use_keep_mask=use_keep_mask,
                )

                weight_norm = weight_norm_sum(model)
                sharpness = (
                    sharpness_proxy(
                        model,
                        test_dl,
                        device,
                        blank_id=blank_id,
                        eps=sharpness_eps,
                        n_batches=sharpness_n_batches,
                        relative_eps=sharpness_relative_eps,
                        use_keep_mask=use_keep_mask,
                    )
                    if sharpness_n_batches > 0
                    else float("nan")
                )
                test_loss_val = test_metrics["loss"]
                sharpness_rel = (
                    sharpness / (float(test_loss_val) + 1e-12)
                    if sharpness_n_batches > 0
                    else float("nan")
                )

                row = {
                    "step": opt_step,
                    "train_loss": train_loss_step_avg,
                    "train_loss_iter_avg": train_loss_iter_avg,
                    "train_eval_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "test_loss": test_loss_val,
                    "train_wer": train_metrics["wer"],
                    "val_wer": val_metrics["wer"],
                    "test_wer": test_metrics["wer"],
                    "weight_norm_sum": weight_norm,
                    "sharpness_proxy": sharpness,
                    "sharpness_rel": sharpness_rel,
                    "generalization_gap": (
                        test_metrics["loss"] - train_metrics["loss"]
                    ),
                    "lr": scheduler.get_last_lr()[0],
                    "skipped_overflow_steps": interval_overflow,
                    "skipped_keep0_batches": interval_keep0,
                    "skipped_nan_batches": interval_nan,
                    "iters_per_opt_step": iters_per_opt_step,
                }
                for key in (
                    "label_noise_mode",
                    "label_noise_p",
                    "label_noise_k",
                    "rt_noise_k",
                    "rt_noise_p",
                ):
                    if key in tc:
                        row[key] = tc[key]
                step_logs.append(row)
                tqdm.write(
                    f"step {opt_step} train_loss={train_loss_step_avg:.4f} "
                    f"train_eval_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"test_loss={test_loss_val:.4f} "
                    f"train_wer={train_metrics['wer']:.4f} "
                    f"val_wer={val_metrics['wer']:.4f} "
                    f"test_wer={test_metrics['wer']:.4f} "
                    f"wnorm={weight_norm:.2f} sharp={sharpness:.3e} "
                    f"sharp_rel={sharpness_rel:.3e} lr={row['lr']:.2e} "
                    f"skip_ovf={interval_overflow} keep0={interval_keep0} "
                    f"iters/step={iters_per_opt_step:.2f}"
                )

                if on_eval_callback is not None:
                    on_eval_callback(opt_step, row, model)

    pbar.close()
    return model, step_logs

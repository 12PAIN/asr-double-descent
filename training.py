"""
Epoch-based training loop for Conformer CTC.
Used by train.py (single run, CLI) and sweep_run.py (multi run, config JSON).
"""

import datetime
import torch
import evaluate
from tqdm import tqdm
import random, numpy as np
from conformer import ConformerCTC
from utils import (
    WarmupCosineScheduler,
    compute_conformer_complexity,
    ctc_lipschitz_proxy,
    evaluate_dataloader,
    rademacher_gen_bound,
    save_final_predictions,
    sharpness_proxy,
    train_one_step,
    weight_norm_sum,
)

DEFAULT_TRAIN_CONFIG = {
    "max_epochs": 30,
    "eval_every_epochs": 1,
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
    "clip_B": 10.0,
    "lipschitz_n_batches": 0,
    "delta": 0.05,
    "C_prime": 4.0,
    "T_max": 750,
    "save_predictions": True,
    "beam_sizes": [5, 10, 50],
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


def build_optimizer_and_scheduler(model, train_config, total_steps):
    """Build AdamW optimizer and LR scheduler. total_steps must be pre-computed."""
    tc = {**DEFAULT_TRAIN_CONFIG, **train_config}
    lr = tc["lr"]
    weight_decay = tc["weight_decay"]
    max_lr = tc["max_lr"]
    scheduler_name = tc["scheduler"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if scheduler_name == "one_cycle":
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
    else:
        warmup_steps = max(1, int(0.05 * total_steps))
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
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
    progress_desc="Training",
    on_eval_callback=None,
    val_dl=None,
    n_train=0,
    output_dir=None,
):
    """
    Epoch-based training loop. Evaluates at the end of every eval_every_epochs epochs.

    Args:
        model_config: dict (d_model, n_heads, num_layers, dropout).
        train_config: dict (max_epochs, eval_every_epochs, scheduler, lr, ...).
        train_dl, train_eval_dl, test_dl: DataLoaders.
        sp, blank_id, vocab_size: tokenizer and vocab.
        device: torch device.
        seed: optional; sets torch/numpy/random seeds before model construction.
        on_eval_callback: optional callable(epoch, row, model) after each eval.
        output_dir: if set and save_predictions=True, writes final prediction JSON files.

    Returns:
        (model, epoch_logs) - epoch_logs is a list of dicts, one per eval epoch.
    """
    tc = {**DEFAULT_TRAIN_CONFIG, **train_config}
    max_epochs = tc["max_epochs"]
    eval_every_epochs = tc["eval_every_epochs"]
    max_eval_batches = tc["max_eval_batches"]
    sharpness_n_batches = tc["sharpness_n_batches"]
    sharpness_eps = tc["sharpness_eps"]
    sharpness_relative_eps = tc["sharpness_relative_eps"]
    use_keep_mask = tc.get("use_keep_mask", False)
    clip_B = tc.get("clip_B", 10.0)
    lipschitz_n_batches = tc.get("lipschitz_n_batches", 0)
    delta = tc.get("delta", 0.05)
    C_prime = tc.get("C_prime", 4.0)
    T_max = tc.get("T_max", 750)
    # P(label corrupted) <= p_static + p_rt (union bound; cap at 1)
    noise_p = min(tc.get("label_noise_p", 0.0) + tc.get("rt_noise_p", 0.0), 1.0)
    annotation_bias_bound = noise_p * clip_B

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = build_model(model_config, vocab_size, device)
    steps_per_epoch = len(train_dl)
    total_steps = max_epochs * steps_per_epoch
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, train_config, total_steps
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    ctc = torch.nn.CTCLoss(
        blank=blank_id, reduction="mean", zero_infinity=True
    ).to(device)

    epoch_logs = []
    opt_step = 0

    pbar = tqdm(total=total_steps, desc=progress_desc)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss_step = 0.0
        running_n_step = 0
        running_loss_iter = 0.0
        running_n_iter = 0
        iters_in_epoch = 0
        opt_steps_in_epoch = 0
        skipped_overflow = 0
        skipped_keep0 = 0
        skipped_nan = 0

        for batch in train_dl:
            iters_in_epoch += 1
            loss_val, did_step, skip_reason = train_one_step(
                model, batch, optimizer, scaler, device,
                blank_id, use_amp, ctc, use_keep_mask=use_keep_mask,
            )
            if skip_reason == "overflow":
                skipped_overflow += 1
            elif skip_reason == "keep0":
                skipped_keep0 += 1
            elif skip_reason == "nan":
                skipped_nan += 1

            if did_step and loss_val is not None:
                running_loss_step += loss_val
                running_n_step += 1
            if loss_val is not None:
                running_loss_iter += loss_val
                running_n_iter += 1

            if did_step:
                opt_step += 1
                opt_steps_in_epoch += 1
                scheduler.step()
                pbar.update(1)

        # end of epoch - eval if due
        if epoch % eval_every_epochs == 0:
            train_loss_step_avg = (
                running_loss_step / running_n_step if running_n_step else float("nan")
            )
            train_loss_iter_avg = (
                running_loss_iter / running_n_iter if running_n_iter else float("nan")
            )
            iters_per_opt_step = (
                iters_in_epoch / opt_steps_in_epoch if opt_steps_in_epoch else float("nan")
            )

            train_metrics = evaluate_dataloader(
                model, train_eval_dl, sp, device,
                blank_id=blank_id, wer_metric=wer_metric, cer_metric=cer_metric,
                max_batches=max_eval_batches, use_amp=use_amp,
                desc=f"[E{epoch}] Train eval", return_texts=True,
                use_keep_mask=use_keep_mask, clip_B=clip_B,
            )
            if val_dl is not None:
                val_metrics = evaluate_dataloader(
                    model, val_dl, sp, device,
                    blank_id=blank_id, wer_metric=wer_metric, cer_metric=cer_metric,
                    max_batches=max_eval_batches, use_amp=use_amp,
                    desc=f"[E{epoch}] Val eval", return_texts=True,
                    use_keep_mask=use_keep_mask, clip_B=clip_B,
                )
            else:
                val_metrics = {
                    "loss": float("nan"), "wer": float("nan"), "cer": float("nan"),
                    "norm_loss": float("nan"), "clipped_norm_loss": float("nan"),
                    "alignment_entropy": float("nan"), "blank_mass": float("nan"),
                    "seq_confidence": float("nan"),
                }
            test_metrics = evaluate_dataloader(
                model, test_dl, sp, device,
                blank_id=blank_id, wer_metric=wer_metric, cer_metric=cer_metric,
                max_batches=max_eval_batches, use_amp=use_amp,
                desc=f"[E{epoch}] Test eval", return_texts=True,
                use_keep_mask=use_keep_mask, clip_B=clip_B,
            )

            weight_norm = weight_norm_sum(model)
            sharpness = (
                sharpness_proxy(
                    model, test_dl, device, blank_id=blank_id,
                    eps=sharpness_eps, n_batches=sharpness_n_batches,
                    relative_eps=sharpness_relative_eps,
                    use_keep_mask=use_keep_mask, use_amp=use_amp,
                )
                if sharpness_n_batches > 0 else float("nan")
            )
            lipschitz_emp = (
                ctc_lipschitz_proxy(
                    model, test_dl, device, blank_id=blank_id,
                    n_batches=lipschitz_n_batches, use_amp=use_amp,
                )
                if lipschitz_n_batches > 0 else float("nan")
            )
            test_loss_val = test_metrics["loss"]
            sharpness_rel = (
                sharpness / (float(test_loss_val) + 1e-12)
                if sharpness_n_batches > 0 else float("nan")
            )

            _cn_train = train_metrics["clipped_norm_loss"]
            _cn_test = test_metrics["clipped_norm_loss"]
            _norm_gap = _cn_test - _cn_train
            _oracle_upper = _cn_test + annotation_bias_bound

            _C_conf = _C_G = _rad_term = _conf_term = _gen_bound = _oracle_risk_bound = float("nan")
            if n_train > 0:
                try:
                    _cpx = compute_conformer_complexity(
                        model, n_train=n_train, T_max=T_max, K=vocab_size
                    )
                    _C_conf = _cpx["C_conf"]
                    _C_G = _cpx["C_G"]
                    _bnd = rademacher_gen_bound(
                        C_G=_C_G, n=n_train, B=clip_B, delta=delta, C_prime=C_prime
                    )
                    _rad_term = _bnd["rademacher_term"]
                    _conf_term = _bnd["confidence_term"]
                    _gen_bound = _bnd["gen_bound"]
                    _oracle_risk_bound = _cn_train + _gen_bound + annotation_bias_bound
                except Exception as _e:
                    tqdm.write(f"[bound] compute_conformer_complexity failed: {_e}")

            _now = datetime.datetime.now(datetime.timezone.utc)
            row = {
                "epoch": epoch,
                "step": opt_step,
                "eval_timestamp": _now.isoformat(),
                "train_loss": train_loss_step_avg,
                "train_loss_iter_avg": train_loss_iter_avg,
                "train_eval_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "test_loss": test_loss_val,
                "train_wer": train_metrics["wer"],
                "val_wer": val_metrics["wer"],
                "test_wer": test_metrics["wer"],
                "train_cer": train_metrics["cer"],
                "val_cer": val_metrics.get("cer", float("nan")),
                "test_cer": test_metrics["cer"],
                "train_alignment_entropy": train_metrics["alignment_entropy"],
                "val_alignment_entropy":   val_metrics.get("alignment_entropy", float("nan")),
                "test_alignment_entropy":  test_metrics["alignment_entropy"],
                "train_blank_mass":  train_metrics["blank_mass"],
                "val_blank_mass":    val_metrics.get("blank_mass", float("nan")),
                "test_blank_mass":   test_metrics["blank_mass"],
                "train_seq_confidence": train_metrics["seq_confidence"],
                "val_seq_confidence":   val_metrics.get("seq_confidence", float("nan")),
                "test_seq_confidence":  test_metrics["seq_confidence"],
                "weight_norm_sum": weight_norm,
                "sharpness_proxy": sharpness,
                "sharpness_rel": sharpness_rel,
                "generalization_gap": test_metrics["loss"] - train_metrics["loss"],
                "norm_train_eval_loss": train_metrics["norm_loss"],
                "norm_test_loss": test_metrics["norm_loss"],
                "clipped_norm_train_eval_loss": _cn_train,
                "clipped_norm_test_loss": _cn_test,
                "norm_generalization_gap": _norm_gap,
                "annotation_bias_bound": annotation_bias_bound,
                "oracle_risk_upper": _oracle_upper,
                "clip_B": clip_B,
                "lipschitz_proxy_empirical": lipschitz_emp,
                "C_conf": _C_conf,
                "C_G": _C_G,
                "rademacher_term": _rad_term,
                "confidence_term": _conf_term,
                "gen_bound": _gen_bound,
                "oracle_risk_bound": _oracle_risk_bound,
                "lr": scheduler.get_last_lr()[0],
                "skipped_overflow_steps": skipped_overflow,
                "skipped_keep0_batches": skipped_keep0,
                "skipped_nan_batches": skipped_nan,
                "iters_per_opt_step": iters_per_opt_step,
            }
            for key in (
                "label_noise_mode", "label_noise_p", "label_noise_k",
                "rt_noise_k", "rt_noise_p",
            ):
                if key in tc:
                    row[key] = tc[key]
            epoch_logs.append(row)
            tqdm.write(
                f"[{_now.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
                f"epoch {epoch}/{max_epochs} step {opt_step} "
                f"train_loss={train_loss_step_avg:.4f} "
                f"train_eval_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"test_loss={test_loss_val:.4f} "
                f"train_wer={train_metrics['wer']:.4f} "
                f"val_wer={val_metrics['wer']:.4f} "
                f"test_wer={test_metrics['wer']:.4f} "
                f"wnorm={weight_norm:.2f} sharp={sharpness:.3e} "
                f"lr={row['lr']:.2e} skip_ovf={skipped_overflow}"
            )

            if on_eval_callback is not None:
                on_eval_callback(epoch, row, model)

    pbar.close()

    if tc.get("save_predictions", False) and output_dir:
        beam_sizes = tc.get("beam_sizes", [10])
        save_final_predictions(
            model, val_dl, test_dl, sp, device, blank_id,
            output_dir=output_dir, beam_sizes=beam_sizes,
            wer_metric=wer_metric, cer_metric=cer_metric,
            use_amp=use_amp, use_keep_mask=use_keep_mask,
        )

    return model, epoch_logs

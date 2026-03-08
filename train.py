#!/usr/bin/env python3
"""Train Conformer CTC on LibriSpeech: single run from CLI (uses shared training loop)."""
import argparse
import json
import os

import torch

from data import load_librispeech
from training import count_parameters, run_training


def main():
    p = argparse.ArgumentParser(
        description="Train Conformer CTC on LibriSpeech"
    )
    p.add_argument("--sp_model", type=str, default="spm_unigram.model")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument(
        "--max_steps",
        type=int,
        default=100_000,
        help="Total optimizer steps (replaces epochs)",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=5000,
        help="Evaluate and log every N steps",
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument(
        "--max_lr",
        type=float,
        default=1e-4,
        help="Max LR (OneCycle or warmup+cosine)",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        choices=("one_cycle", "warmup_cosine"),
        default="warmup_cosine",
        help="LR schedule: warmup_cosine (5%% warmup + cosine) or one_cycle",
    )
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_max_size", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument(
        "--max_eval_batches",
        type=int,
        default=500,
        help="Fixed eval set size (batches); same batches every run (same seed)",
    )
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training set size for quick experiments",
    )
    p.add_argument(
        "--include_val_in_train",
        action="store_true",
        help="Add validation set to training data (default: False for stage A)",
    )
    p.add_argument("--sharpness_n_batches", type=int, default=100)
    p.add_argument("--sharpness_eps", type=float, default=1e-2)
    p.add_argument("--no_sharpness_relative_eps", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--cmvn_mode",
        type=str,
        choices=("none", "utt", "global", "batch"),
        default="utt",
        help="Input normalization: none, utt (per-utterance), global (train stats), batch",
    )
    p.add_argument(
        "--global_cmvn_path",
        type=str,
        default=None,
        help="Path to load/save global CMVN stats (default: global_cmvn_librispeech_clean.pt)",
    )
    p.add_argument(
        "--compute_global_cmvn",
        action="store_true",
        help="Recompute global CMVN from train and save (even if file exists)",
    )
    p.add_argument(
        "--cmvn_max_hours",
        type=float,
        default=None,
        help="Cap hours of audio when computing global CMVN (for speed)",
    )
    p.add_argument(
        "--label_noise_mode",
        type=str,
        choices=("none", "static", "static_disk"),
        default="none",
        help="Label noise: none, static (in-memory), static_disk (cache on disk)",
    )
    p.add_argument("--label_noise_p", type=float, default=0.0)
    p.add_argument("--label_noise_k", type=int, default=0)
    p.add_argument("--label_noise_seed", type=int, default=0)
    p.add_argument(
        "--label_noise_replace",
        type=str,
        choices=("random", "unk"),
        default="random",
    )
    p.add_argument("--label_noise_disk_path", type=str, default=None)
    p.add_argument("--label_noise_use_disk", action="store_true")
    p.add_argument("--label_noise_save_disk", action="store_true")
    p.add_argument("--rt_noise_k", type=int, default=0)
    p.add_argument("--rt_noise_p", type=float, default=0.0)
    p.add_argument("--rt_noise_seed", type=int, default=None)
    p.add_argument(
        "--rt_noise_replace",
        type=str,
        choices=("random", "unk"),
        default="random",
    )
    p.add_argument(
        "--keep_mask",
        action="store_true",
        help="Enable keep = out_lengths >= target_lengths guard (default: disabled). "
             "CTCLoss(zero_infinity=True) handles short seqs without it.",
    )
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_config = {
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "scheduler": args.scheduler,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_lr": args.max_lr,
        "max_eval_batches": args.max_eval_batches,
        "sharpness_n_batches": args.sharpness_n_batches,
        "sharpness_eps": args.sharpness_eps,
        "sharpness_relative_eps": not args.no_sharpness_relative_eps,
        "use_keep_mask": args.keep_mask,
        "label_noise_mode": args.label_noise_mode,
        "label_noise_p": args.label_noise_p,
        "label_noise_k": args.label_noise_k,
        "rt_noise_k": args.rt_noise_k,
        "rt_noise_p": args.rt_noise_p,
    }
    model_config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }

    train_dl, val_dl, test_dl, train_eval_dl, sp, blank_id = load_librispeech(
        args.sp_model,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        val_max_size=args.val_max_size,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        include_val_in_train=args.include_val_in_train,
        max_eval_batches=args.max_eval_batches,
        cmvn_mode=args.cmvn_mode,
        global_cmvn_path=args.global_cmvn_path,
        compute_global_cmvn=args.compute_global_cmvn,
        cmvn_max_hours=args.cmvn_max_hours,
        label_noise_mode=args.label_noise_mode,
        label_noise_p=args.label_noise_p,
        label_noise_k=args.label_noise_k,
        label_noise_seed=args.label_noise_seed,
        label_noise_replace=args.label_noise_replace,
        label_noise_disk_path=args.label_noise_disk_path,
        label_noise_use_disk=args.label_noise_use_disk,
        label_noise_save_disk=args.label_noise_save_disk,
        rt_noise_k=args.rt_noise_k,
        rt_noise_p=args.rt_noise_p,
        rt_noise_seed=args.rt_noise_seed,
        rt_noise_replace=args.rt_noise_replace,
    )
    vocab_size = sp.get_piece_size()
    best_test_wer = float("inf")

    def on_eval(step_1based, row, model):
        nonlocal best_test_wer
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_dir, f"checkpoint_step_{step_1based}.pth"
                ),
            )
            test_wer = row.get("test_wer", float("inf"))
            if isinstance(test_wer, (int, float)) and test_wer < best_test_wer:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "best.pth"),
                )
                best_test_wer = test_wer

    model, step_logs = run_training(
        model_config=model_config,
        train_config=train_config,
        train_dl=train_dl,
        train_eval_dl=train_eval_dl,
        test_dl=test_dl,
        sp=sp,
        blank_id=blank_id,
        vocab_size=vocab_size,
        device=device,
        seed=args.seed,
        progress_desc="Steps",
        on_eval_callback=on_eval,
        val_dl=None if args.include_val_in_train else val_dl,
    )

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(
            model.state_dict(), os.path.join(args.output_dir, "final.pth")
        )
        with open(
            os.path.join(args.output_dir, "train_log.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(step_logs, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

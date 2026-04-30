#!/usr/bin/env python3
"""
Sweep: multiple seeds × model configs. Uses shared training loop from training.py.
"""
import argparse
import hashlib
import json
import os

import torch
from tqdm import tqdm

from data import load_librispeech
from training import count_parameters, run_training


def run_id_from_config(
    seed: int, model_config: dict, train_kwargs: dict
) -> str:
    """Unique string for this run (seed + model + train settings)."""
    key = json.dumps(
        {
            "seed": seed,
            "model": sorted(model_config.items()),
            "train": sorted(train_kwargs.items()),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_results(path: str) -> list:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_results(path: str, results: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description="Sweep: train multiple Conformer configs × seeds (same logic as train.py)"
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--output", "--results", type=str, dest="output", required=True
    )
    p.add_argument("--sp_model", type=str, default="spm_unigram.model")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    seeds = cfg["seeds"]
    model_configs = cfg["model_configs"]
    train_kw = cfg.get("train", {})

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = load_results(args.output)
    done_ids = {r["run_id"] for r in results}

    for seed in seeds:
        for model_config in model_configs:
            run_id = run_id_from_config(seed, model_config, train_kw)
            if run_id in done_ids:
                tqdm.write(
                    f"Skip: run_id={run_id} seed={seed} config={model_config}"
                )
                continue

            train_dl, val_dl, test_dl, train_eval_dl, sp, blank_id = (
                load_librispeech(
                    args.sp_model,
                    batch_size=train_kw.get("batch_size", 16),
                    eval_batch_size=train_kw.get("eval_batch_size"),
                    num_workers=train_kw.get("num_workers", 4),
                    val_max_size=train_kw.get("val_max_size"),
                    seed=seed,
                    max_train_samples=train_kw.get("max_train_samples"),
                    include_val_in_train=train_kw.get(
                        "include_val_in_train", False
                    ),
                    max_eval_batches=train_kw.get("max_eval_batches", 50),
                    cmvn_mode=train_kw.get("cmvn_mode", "utt"),
                    global_cmvn_path=train_kw.get("global_cmvn_path"),
                    compute_global_cmvn=train_kw.get(
                        "compute_global_cmvn", False
                    ),
                    cmvn_max_hours=train_kw.get("cmvn_max_hours"),
                    label_noise_mode=train_kw.get("label_noise_mode", "none"),
                    label_noise_p=train_kw.get("label_noise_p", 0.0),
                    label_noise_k=train_kw.get("label_noise_k", 0),
                    label_noise_seed=train_kw.get("label_noise_seed", 0),
                    label_noise_replace=train_kw.get(
                        "label_noise_replace", "random"
                    ),
                    label_noise_disk_path=train_kw.get("label_noise_disk_path"),
                    label_noise_use_disk=train_kw.get(
                        "label_noise_use_disk", False
                    ),
                    label_noise_save_disk=train_kw.get(
                        "label_noise_save_disk", False
                    ),
                    rt_noise_k=train_kw.get("rt_noise_k", 0),
                    rt_noise_p=train_kw.get("rt_noise_p", 0.0),
                    rt_noise_seed=train_kw.get("rt_noise_seed"),
                    rt_noise_replace=train_kw.get(
                        "rt_noise_replace", "random"
                    ),
                )
            )
            vocab_size = sp.get_piece_size()

            run_output_dir = os.path.join(
                os.path.splitext(args.output)[0], run_id
            ) if train_kw.get("save_predictions", False) else None

            model, epoch_logs = run_training(
                model_config=model_config,
                train_config=train_kw,
                train_dl=train_dl,
                train_eval_dl=train_eval_dl,
                test_dl=test_dl,
                sp=sp,
                blank_id=blank_id,
                vocab_size=vocab_size,
                device=device,
                seed=seed,
                progress_desc=f"seed={seed} L={model_config.get('num_layers')}",
                on_eval_callback=None,
                val_dl=None
                if train_kw.get("include_val_in_train", False)
                else val_dl,
                n_train=len(train_dl.dataset) if hasattr(train_dl, "dataset") else 0,
                output_dir=run_output_dir,
            )

            run_record = {
                "run_id": run_id,
                "seed": seed,
                "model_config": model_config,
                "n_params": count_parameters(model),
                "train_config": train_kw,
                "epochs": epoch_logs,
            }
            results.append(run_record)
            save_results(args.output, results)
            done_ids.add(run_id)
            tqdm.write(
                f"Saved run_id={run_id} seed={seed} config={model_config}"
            )

    print("Sweep done. Results written to", args.output)


if __name__ == "__main__":
    main()

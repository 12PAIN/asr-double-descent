# Pipeline: ASR on LibriSpeech (Conformer + CTC)

Trains and evaluates a Conformer-CTC model on LibriSpeech clean. Experiments cover overparameterization, double descent, regularization, and label noise.

---

## 1. Overview

```
LibriSpeech (HF) -> load_librispeech() -> DataLoaders (train/val/test/train_eval)
       V
CollateCTC: audio -> mel -> log -> CMVN -> (optional runtime label noise)
       V
run_training(): loop over optimizer steps (max_steps), eval every eval_every
       V
step_logs (train/val/test loss, WER, sharpness, ...) + checkpoints / sweep JSON
```

- **Data**: HuggingFace `openslr/librispeech_asr`, clean (train.360 + train.100, validation, test). Audio at 16 kHz; text tokenized with SentencePiece; collate produces 80-dim log-mel features (same as Conformer `in_feats=80`).
- **Training**: Step-based (not epoch-based). One step = one batch -> forward -> CTC loss -> backward -> optimizer.step(); steps can be skipped on AMP overflow (counters are logged).
- **Entry points**: `train.py` (single run from CLI), `sweep_run.py` (multiple configs × seeds from JSON).

---

## 2. Data and collate (`data.py`)

### 2.1. Loading: `load_librispeech(...)`

Returns: `train_dl`, `val_dl`, `test_dl`, `train_eval_dl`, `sp` (SentencePiece), `blank_id`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sp_model_path` | - | Path to SentencePiece model (must include `<blank>`). |
| `batch_size` | 32 | Batch size for train_dl. |
| `eval_batch_size` | = batch_size | Batch size for val/test/train_eval. |
| `num_workers` | 8 | Workers for train_dl only (saves RAM on full dataset). |
| `eval_num_workers` | 0 | Workers for val/test/train_eval (0 = no extra processes). |
| `val_max_size` | None | Cap number of validation examples (for quick runs). |
| `seed` | 0 | Seed for shuffle and worker_init_fn (reproducibility). |
| `max_train_samples` | None | Use only first N training examples (quick experiments). |
| `include_val_in_train` | False | Whether to add val to train (Stage A default: no val to avoid data leakage). |
| `max_eval_batches` | 50 | Number of batches for train_eval_dl (fixed subset of train for train_eval loss). |

**CMVN (feature normalization):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cmvn_mode` | `"utt"` | Mode: `"none"`, `"utt"` (per-utterance), `"global"` (train statistics), `"batch"` (per batch). |
| `global_cmvn_path` | None | Path to mean/std file for global CMVN; if None, uses `global_cmvn_librispeech_clean.pt`. |
| `compute_global_cmvn` | False | Recompute global CMVN from train and save (even if file exists). |
| `cmvn_max_hours` | None | Cap on hours of audio when computing global CMVN (speed). |
| `cmvn_eps` | 1e-5 | Epsilon for std in normalization. |

**Static label noise (train only):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `label_noise_mode` | `"none"` | `"none"` = no noise; `"static"` = corrupt once in memory; `"static_disk"` = cache on disk. |
| `label_noise_p` | 0.0 | Probability to corrupt a token (when k=0). |
| `label_noise_k` | 0 | Number of tokens to corrupt per example (if >0, overrides p). |
| `label_noise_seed` | 0 | Seed for deterministic corruption (seed + example index). |
| `label_noise_replace` | `"random"` | Replacement: `"random"` (uniform over vocab excluding blank) or `"unk"`. |
| `label_noise_disk_path` | None | Path to .pt cache of corrupted targets (default `static_label_noise.pt`). |
| `label_noise_use_disk` | False | Try to load cache from disk. |
| `label_noise_save_disk` | False | Save corrupted targets to disk after computing. |

**Runtime label noise (in collate, every batch):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rt_noise_k` | 0 | Number of tokens to corrupt per example in collate (if >0, overrides p). |
| `rt_noise_p` | 0.0 | Probability to corrupt a token in collate. |
| `rt_noise_seed` | None | If set, deterministic noise (seed + collate call counter). |
| `rt_noise_replace` | `"random"` | `"random"` or `"unk"`. |

Constraints: `blank_id` never appears in target sequences (static or runtime); `replace="unk"` requires a valid `<unk>` in the SPM model.

---

### 2.2. Collate `CollateCTC`

Input: list of examples with `audio` (array), `targets` (list of token ids).
Output: `(features, feature_lengths, targets, target_lengths)` - tensors for CTC.

**Steps:**

1. Audio -> MelSpectrogram (16k, 80 mel, n_fft=400, hop=160, win=400) -> `log(mel.clamp_min(1e-10))`.
2. CMVN by selected mode: none / utt (mean/std over time) / global (loaded mean, std) / batch (mean/std over batch after padding).
3. Padding in time and in target length (pad_id = blank_id).
4. Optionally: runtime label noise on assembled `targets` and `target_lengths` (only within valid length).

Collate parameters are set when created in `load_librispeech` (cmvn_mode, global_stats, eps, rt_noise_*, vocab_size, unk_id).

---

## 3. Training (`training.py`)

### 3.1. `run_training(model_config, train_config, train_dl, train_eval_dl, test_dl, sp, blank_id, vocab_size, device, seed=..., progress_desc=..., on_eval_callback=..., val_dl=None)`

- Builds ConformerCTC and optimizer/scheduler from configs.
- Loops over **optimizer steps** (not batch iterations): `opt_step` increments only when `train_one_step` returns `did_step=True`, so the scheduler and logs match exactly `max_steps` steps.
- Every `eval_every` steps and on the last step: evaluate on train_eval_dl, val_dl (if provided), and test_dl (loss + WER), sharpness proxy, weight norm; append a row to `step_logs` and call `on_eval_callback(opt_step, row, model)` if provided.
- `val_dl`: optional validation DataLoader. When provided, `val_loss` and `val_wer` are computed each eval; when None (default, Stage A), they are logged as NaN.

**model_config** (architecture):

| Key | Default | Description |
|-----|---------|-------------|
| `d_model` | 512 | Model dimension. |
| `n_heads` | 8 | Number of attention heads. |
| `num_layers` | 16 | Number of Conformer layers. |
| `dropout` | 0.0 | Dropout. |

**train_config** (training and evaluation):

| Key | Default | Description |
|-----|---------|-------------|
| `max_steps` | 100_000 | Total optimizer steps. |
| `eval_every` | 5000 | Evaluate and log every N steps. |
| `scheduler` | `"warmup_cosine"` | `"warmup_cosine"` (5% warmup + cosine) or `"one_cycle"`. |
| `batch_size` | 16 | For log compatibility (actual batch set in load_librispeech). |
| `eval_batch_size` | None | Same. |
| `lr` | 5e-5 | Initial/base learning rate. |
| `weight_decay` | 1e-6 | AdamW weight decay. |
| `max_lr` | 1e-4 | Peak LR (warmup_cosine / one_cycle). |
| `max_eval_batches` | 50 | Max batches when evaluating on test (and train_eval). |
| `sharpness_n_batches` | 10 | Batches for sharpness proxy (CLI default: 100). |
| `sharpness_eps` | 1e-3 | Perturbation epsilon for sharpness. |
| `sharpness_relative_eps` | True | Relative perturbation (w.r.t. weight norm). |
| `use_keep_mask` | False | Enable keep = out_lengths >= target_lengths guard. When False (default), CTCLoss(zero_infinity=True) handles short sequences without a GPU→CPU sync. |

If present in `train_config`, these are also added to each log row: `label_noise_mode`, `label_noise_p`, `label_noise_k`, `rt_noise_k`, `rt_noise_p`.

**Fields in each step_logs entry (per eval):**

- `step` - optimizer step number.
- `train_loss` - average loss over **steps** (only when a step was taken).
- `train_loss_iter_avg` - average loss over batch iterations (including skipped).
- `train_eval_loss`, `val_loss`, `test_loss` - CTC loss on train_eval, validation, and test.
- `train_wer`, `val_wer`, `test_wer` - WER on each split (`val_*` is NaN when val_dl is None).
- `weight_norm_sum` - L2 norm of weights.
- `sharpness_proxy`, `sharpness_rel` - sharpness and relative (to test loss).
- `generalization_gap` - test_loss − train_eval_loss.
- `lr` - current learning rate.
- `skipped_overflow_steps`, `skipped_keep0_batches`, `skipped_nan_batches` - counts of skipped steps (AMP overflow, empty batch by keep, nan loss).
- `iters_per_opt_step` - average number of batch iterations per optimizer step over the interval.

---

## 4. Entry points

### 4.1. `train.py`

Single run: parses arguments -> builds `train_config` and `model_config` -> calls `load_librispeech` and `run_training` -> saves checkpoints and `train_log.json`.

**Main CLI flags (all parameters from sections 2 and 3 are available):**

- Data: `--sp_model`, `--batch_size`, `--eval_batch_size`, `--num_workers`, `--seed`, `--max_train_samples`, `--max_eval_batches`, `--include_val_in_train`, `--val_max_size`.
- CMVN: `--cmvn_mode`, `--global_cmvn_path`, `--compute_global_cmvn`, `--cmvn_max_hours`.
- Label noise: `--label_noise_mode`, `--label_noise_p`, `--label_noise_k`, `--label_noise_seed`, `--label_noise_replace`, `--label_noise_disk_path`, `--label_noise_use_disk`, `--label_noise_save_disk`, `--rt_noise_k`, `--rt_noise_p`, `--rt_noise_seed`, `--rt_noise_replace`.
- Training: `--max_steps`, `--eval_every`, `--scheduler`, `--lr`, `--weight_decay`, `--max_lr`.
- Model: `--d_model`, `--n_heads`, `--num_layers`, `--dropout`.
- Sharpness: `--sharpness_n_batches` (default 100), `--sharpness_eps` (default 1e-2), `--no_sharpness_relative_eps`.
- Keep mask: `--keep_mask` (flag; enables the out_lengths >= target_lengths guard and GPU→CPU sync; off by default).
- Other: `--output_dir`, `--device`.

With `--output_dir`: on each eval `checkpoint_step_{step}.pth` is saved; when test_wer improves, `best.pth` is saved; at the end, `final.pth` and `train_log.json` (list of step_logs) are written.

---

### 4.2. `sweep_run.py`

Multiple runs: reads JSON with `seeds`, `model_configs`, `train`; for each (seed, model_config) computes `run_id` (hash of seed + model_config + train); skips if run_id already in results; otherwise loads data with that seed, runs `run_training`, appends to results, and saves the JSON.

**Arguments:**

- `--config` - path to sweep JSON config.
- `--output` - path to output JSON with results.
- `--sp_model`, `--device` - same as train.

**Sweep config structure:**

```json
{
  "seeds": [0, 1],
  "model_configs": [
    { "d_model": 512, "n_heads": 8, "num_layers": 16, "dropout": 0 }
  ],
  "train": {
    "max_steps": 100000,
    "eval_every": 5000,
    "scheduler": "warmup_cosine",
    "batch_size": 16,
    "eval_batch_size": 32,
    "lr": 5e-5,
    "weight_decay": 1e-6,
    "max_lr": 1e-4,
    "num_workers": 4,
    "val_max_size": null,
    "max_eval_batches": 500,
    "max_train_samples": null,
    "include_val_in_train": false,
    "sharpness_n_batches": 100,
    "sharpness_eps": 0.001,
    "sharpness_relative_eps": true,
    "cmvn_mode": "utt",
    "global_cmvn_path": null,
    "label_noise_mode": "none",
    "label_noise_p": 0.0,
    "label_noise_k": 0,
    "label_noise_seed": 0,
    "label_noise_replace": "random",
    "rt_noise_k": 0,
    "rt_noise_p": 0.0,
    "rt_noise_seed": null,
    "rt_noise_replace": "random"
  }
}
```

All keys under `train` are passed to `load_librispeech` and `run_training`; they affect `run_id`, so different label-noise or data settings produce separate run IDs and never share results.

---

## 5. Helper modules

- **`utils.py`**: `train_one_step` (forward, CTC, backward, AMP; returns loss, did_step, skip_reason), `evaluate_dataloader` (loss + WER with optional keep mask), `sharpness_proxy` (one-sided, K perturbations per batch, on-device perturb/restore), `weight_norm_sum`, `WarmupCosineScheduler`, `compute_wer`.
- **`conformer.py`**: ConformerCTC architecture; input 80 mel, output logits over vocabulary.
- **`dataset.py`**: Text normalization for SPM (e.g. lowercasing, stripping extra spaces).
- **`train_spm.py`**: Trains a SentencePiece unigram model from LibriSpeech transcripts; outputs `spm_unigram.model` and `spm_unigram.vocab`.
- **`plot_stage_a.py`**: Plots double-descent / generalization curves from sweep output JSONs (Stage A experiments varying model depth).
- **`plot_cmvn_comparison.py`**: Compares training curves across CMVN modes (`none`, `utt`, `global`) from their respective sweep output JSONs.

---

## 6. Files and artifacts

| File / artifact | Description |
|-----------------|-------------|
| `train_log.json` | List of step_logs entries (single train.py run). |
| `checkpoint_step_*.pth`, `best.pth`, `final.pth` | Model state_dict (when output_dir is set). |
| `global_cmvn_librispeech_clean.pt` | Global CMVN statistics (mean, std over train) when cmvn_mode=global and path not overridden. |
| `static_label_noise.pt` | Cache of statically corrupted targets (when label_noise with save to disk is used). |
| Sweep output (e.g. `outputs/outputs_full.json`) | List of objects: run_id, seed, model_config, n_params, train_config, steps (that run's step_logs). |
| `outputs/` | Directory for sweep result JSONs (e.g. outputs_full.json, outputs_stage_b_p05.json, etc.). |
| `configs/` | Sweep config JSONs for all experiments (see Section 8). |
| `stage_a_plots/`, `cmvn_comparison/` | Directories for saved plot figures. |

---

## 7. Implementation notes

- **Steps vs iterations**: Counting is by actual optimizer steps; on AMP overflow or keep.sum()==0 the step is not taken but the batch is consumed - reflected in iters_per_opt_step and skipped_*.
- **Keep mask**: When `use_keep_mask=False` (default), the `out_lengths >= target_lengths` filter is not applied; `CTCLoss(zero_infinity=True)` handles invalid sequences silently and avoids the GPU→CPU sync. Enable with `--keep_mask` if explicit filtering is needed.
- **Validation eval**: `val_dl` is passed to `run_training` when `include_val_in_train=False`; Stage A passes `val_dl=None` to keep val out of training.
- **Reproducibility**: Set seed in load_librispeech (shuffle, workers) and in run_training (torch/np/random); for static noise use label_noise_seed; for runtime use rt_noise_seed when set.
- **Stage A**: `include_val_in_train=False` by default; `train.py` passes `val_dl` only when val is separate from train.

---

## 8. Configs (`configs/`)

All configs are JSON files consumed by `sweep_run.py`. Each defines `seeds`, `model_configs`, and `train` settings.

| Config file | Purpose |
|-------------|---------|
| `sweep_full.json` | **Stage A baseline**: 8 model depths (num_layers 1–16), 2 seeds, full train set, `cmvn_mode="utt"`. Used to study double descent vs model size. |
| `sweep_full_seed_0.json` | Same as `sweep_full.json` but single seed (0). |
| `sweep_subset_5k.json` | Stage A on a **5k-sample subset**: same 8 depths, 2 seeds, `max_train_samples=5000`. Exaggerates overfitting for double-descent visibility. |
| `sweep_subset_50k.json` | Stage A on a **50k-sample subset**: same 8 depths, 2 seeds, `max_train_samples=50000`. |
| `sweep_subset_150k.json` | Stage A on a **150k-sample subset**: same 8 depths, 2 seeds, `max_train_samples=150000`. |
| `sweep_regularization_none.json` | **CMVN ablation — no normalization**: single depth (8 layers), 2 seeds, `cmvn_mode="none"`. |
| `sweep_regularization_utt.json` | **CMVN ablation — per-utterance**: single depth (8 layers), 2 seeds, `cmvn_mode="utt"`. |
| `sweep_regularization_global.json` | **CMVN ablation — global stats**: single depth (8 layers), 2 seeds, `cmvn_mode="global"`. Uses `global_cmvn_librispeech_clean.pt`. |
| `sweep_stage_b_p05.json` | **Stage B — label noise p=0.05**: 3 depths (4/8/16 layers), 2 seeds, `label_noise_mode="static"`, `label_noise_p=0.05`, seed 42. |
| `sweep_stage_b_p10.json` | **Stage B — label noise p=0.10**: same 3 depths, 2 seeds, `label_noise_p=0.10`. |
| `sweep_stage_b_p20.json` | **Stage B — label noise p=0.20**: same 3 depths, 2 seeds, `label_noise_p=0.20`. |
| `sweep_test_1.json` | Quick smoke-test config (small run for debugging). |
| `sweep_test_2.json` | Quick smoke-test config (small run for debugging). |

**Common train settings across all experiment configs:**

- `max_steps=100000`, `eval_every=5000`, `scheduler="warmup_cosine"`.
- `batch_size=16`, `eval_batch_size=32`, `lr=5e-5`, `max_lr=1e-4`, `weight_decay=1e-6`.
- `max_eval_batches=500`, `sharpness_n_batches=100`, `sharpness_eps=1e-3`, `sharpness_relative_eps=true`.
- `include_val_in_train=false` (no data leakage).

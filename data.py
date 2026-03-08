"""Data loading and collation for LibriSpeech CTC training."""

import hashlib
import json
import os
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader
import sentencepiece as spm
from datasets import load_dataset, Audio, concatenate_datasets
from dataset import normalize_text
import random
import numpy as np

# Static label-noise cache version; bump when format changes
STATIC_NOISE_CACHE_VERSION = 1


def _corrupt_targets_static(
    tokens: List[int],
    idx: int,
    seed: int,
    noise_p: float,
    noise_k: int,
    replace: str,
    vocab_size: int,
    blank_id: int,
    unk_id: int,
) -> List[int]:
    """Corrupt a single transcript deterministically. Never insert blank_id."""
    if not tokens:
        return tokens
    rng = np.random.default_rng(seed + idx)
    L = len(tokens)
    out = list(tokens)
    if noise_k > 0:
        n_corrupt = min(noise_k, L)
        positions = rng.choice(L, size=n_corrupt, replace=False)
    else:
        positions = [i for i in range(L) if rng.random() < noise_p]
    valid_ids = [i for i in range(vocab_size) if i != blank_id]
    for pos in positions:
        if replace == "unk":
            out[pos] = unk_id
        else:
            candidates = [v for v in valid_ids if v != out[pos]]
            out[pos] = rng.choice(candidates) if candidates else out[pos]
    assert blank_id not in out, "blank_id must not appear in targets"
    return out


def _static_noise_config_hash(
    dataset_fingerprint: str,
    seed: int,
    noise_p: float,
    noise_k: int,
    replace: str,
    vocab_size: int,
) -> str:
    h = hashlib.sha256(
        json.dumps(
            {
                "version": STATIC_NOISE_CACHE_VERSION,
                "fingerprint": dataset_fingerprint,
                "seed": seed,
                "noise_p": noise_p,
                "noise_k": noise_k,
                "replace": replace,
                "vocab_size": vocab_size,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    return h


def load_static_noise_cache(
    path: str,
    dataset_len: int,
    dataset_fingerprint: str,
    seed: int,
    noise_p: float,
    noise_k: int,
    replace: str,
    vocab_size: int,
) -> Optional[List[List[int]]]:
    """Load corrupted targets from disk. Returns None if invalid or missing."""
    if not path or not os.path.isfile(path):
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if d.get("version") != STATIC_NOISE_CACHE_VERSION:
        return None
    if d.get("dataset_fingerprint") != dataset_fingerprint:
        return None
    if len(d["targets_corrupt"]) != dataset_len:
        return None
    cfg = d.get("config", {})
    if (
        cfg.get("seed") != seed
        or cfg.get("noise_p") != noise_p
        or cfg.get("noise_k") != noise_k
        or cfg.get("replace") != replace
        or cfg.get("vocab_size") != vocab_size
    ):
        return None
    return d["targets_corrupt"]


def save_static_noise_cache(
    path: str,
    targets_corrupt: List[List[int]],
    dataset_fingerprint: str,
    seed: int,
    noise_p: float,
    noise_k: int,
    replace: str,
    vocab_size: int,
) -> None:
    """Save corrupted targets only (no audio)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    config = {
        "seed": seed,
        "noise_p": noise_p,
        "noise_k": noise_k,
        "replace": replace,
        "vocab_size": vocab_size,
    }
    torch.save(
        {
            "version": STATIC_NOISE_CACHE_VERSION,
            "dataset_fingerprint": dataset_fingerprint,
            "config": config,
            "targets_corrupt": targets_corrupt,
        },
        path,
    )


def encode_example(ex, sp):
    """Encode text to token ids using SentencePiece; expects 'text' in ex."""
    text = normalize_text(ex["text"])
    ids = sp.encode(text, out_type=int)
    return {"targets": ids}


def _make_mel_transform(
    sample_rate=16000,
    n_mels=80,
    n_fft=400,
    hop_length=160,
    win_length=400,
):
    """Same transform as CollateCTC for reproducible global CMVN."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )


class CollateCTC:
    def __init__(
        self,
        pad_id: int,
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        win_length=400,
        cmvn_mode: str = "utt",
        global_stats: Optional[dict] = None,
        eps: float = 1e-5,
        rt_noise_k: int = 0,
        rt_noise_p: float = 0.0,
        rt_noise_seed: Optional[int] = None,
        rt_noise_replace: str = "random",
        vocab_size: Optional[int] = None,
        unk_id: Optional[int] = None,
    ):
        self.pad_id = pad_id
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.cmvn_mode = cmvn_mode
        self.eps = eps
        self.rt_noise_k = rt_noise_k
        self.rt_noise_p = rt_noise_p
        self.rt_noise_seed = rt_noise_seed
        self.rt_noise_replace = rt_noise_replace
        self.vocab_size = vocab_size
        self.unk_id = unk_id if unk_id is not None else 0
        self.mel = _make_mel_transform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        if global_stats is not None:
            self.global_mean = global_stats["mean"].float()  # [n_mels]
            self.global_std = global_stats["std"].float()  # [n_mels]
        else:
            self.global_mean = None
            self.global_std = None

    def _normalize_utt(self, mel):
        # mel: [1, n_mels, time]
        mean_t = mel.mean(dim=-1, keepdim=True)
        std_t = mel.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (mel - mean_t) / std_t

    def _normalize_global(self, mel):
        # mel: [1, n_mels, time]; global_mean/std: [n_mels]
        m = self.global_mean.view(1, -1, 1)
        s = self.global_std.view(1, -1, 1)
        return (mel - m) / s.clamp_min(self.eps)

    def _apply_runtime_noise(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        gen: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Corrupt targets in-place within valid length. Never insert blank_id."""
        B, Smax = targets.shape
        blank_id = self.pad_id
        vs = self.vocab_size
        unk_id = self.unk_id
        if vs is None or (self.rt_noise_k <= 0 and self.rt_noise_p <= 0.0):
            return targets
        valid_ids = [i for i in range(vs) if i != blank_id]
        if not valid_ids:
            return targets
        for i in range(B):
            L = int(target_lengths[i].item())
            if L <= 0:
                continue
            if self.rt_noise_k > 0:
                n_corrupt = min(self.rt_noise_k, L)
                pos = torch.randperm(L, generator=gen, device=targets.device)[:n_corrupt]
            else:
                mask = torch.rand(L, generator=gen, device=targets.device) < self.rt_noise_p
                pos = torch.where(mask)[0]
            for p in pos:
                p = int(p.item())
                if self.rt_noise_replace == "unk":
                    targets[i, p] = unk_id
                else:
                    idx = int(
                        torch.randint(
                            len(valid_ids), (1,), generator=gen, device=targets.device
                        ).item()
                    )
                    new_id = valid_ids[idx]
                    if new_id == targets[i, p].item() and len(valid_ids) > 1:
                        new_id = valid_ids[(idx + 1) % len(valid_ids)]
                    targets[i, p] = new_id
        for i in range(B):
            valid_len = int(target_lengths[i].item())
            if valid_len > 0:
                assert (
                    targets[i, :valid_len] != blank_id
                ).all(), "blank_id must not appear in valid target region"
        return targets

    def __call__(self, batch):
        feats, feat_lens, targs, targ_lens = [], [], [], []

        for ex in batch:
            wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
            if wav.dim() == 2:
                wav = wav.mean(0)
            wav = wav.unsqueeze(0)

            mel = self.mel(wav)
            mel = torch.log(mel.clamp_min(1e-10))

            if self.cmvn_mode == "none":
                pass
            elif self.cmvn_mode == "utt":
                mel = self._normalize_utt(mel)
            elif self.cmvn_mode == "global":
                mel = self._normalize_global(mel)
            elif self.cmvn_mode == "batch":
                pass  # normalize after stacking
            else:
                raise ValueError(f"Unknown cmvn_mode: {self.cmvn_mode}")

            mel = mel.squeeze(0).transpose(0, 1)  # [time, n_mels]
            feats.append(mel)
            feat_lens.append(mel.size(0))

            ids = torch.tensor(ex["targets"], dtype=torch.long)
            targs.append(ids)
            targ_lens.append(ids.numel())

        B = len(feats)
        F = feats[0].size(1)
        Tmax = max(feat_lens)

        features = torch.zeros(B, Tmax, F, dtype=torch.float32)
        for i, x in enumerate(feats):
            features[i, : x.size(0)] = x

        if self.cmvn_mode == "batch":
            mean = features.mean(dim=(0, 1))
            std = features.std(dim=(0, 1)).clamp_min(self.eps)
            features = (features - mean) / std

        feature_lengths = torch.tensor(feat_lens, dtype=torch.long)

        Smax = max(targ_lens) if targ_lens else 0
        targets = torch.full((B, Smax), fill_value=self.pad_id, dtype=torch.long)
        for i, y in enumerate(targs):
            targets[i, : y.numel()] = y

        target_lengths = torch.tensor(targ_lens, dtype=torch.long)

        if (self.rt_noise_k > 0 or self.rt_noise_p > 0.0) and self.vocab_size:
            gen = None
            if self.rt_noise_seed is not None:
                if not hasattr(self, "_rt_noise_call_count"):
                    self._rt_noise_call_count = 0
                self._rt_noise_call_count += 1
                gen = torch.Generator().manual_seed(self.rt_noise_seed + self._rt_noise_call_count)
            self._apply_runtime_noise(targets, target_lengths, gen=gen)

        return features, feature_lengths, targets, target_lengths


def compute_global_cmvn_stats(
    dataset,
    sample_rate=16000,
    n_mels=80,
    n_fft=400,
    hop_length=160,
    win_length=400,
    max_items: Optional[int] = None,
    max_hours: Optional[float] = None,
    eps: float = 1e-5,
    save_path: Optional[str] = None,
):
    """
    Compute global mean/std over log-mel frames from training set only.
    Streaming, memory-safe. Uses same mel config as CollateCTC.
    Returns dict with "mean" and "std" tensors shape [n_mels].
    """
    mel_fn = _make_mel_transform(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    n_mels_out = n_mels
    sum_f = torch.zeros(n_mels_out, dtype=torch.float64)
    sumsq_f = torch.zeros(n_mels_out, dtype=torch.float64)
    count = 0.0
    max_frames = None
    if max_hours is not None:
        # 16k sr, hop 160 -> 100 frames per second
        max_frames = int(max_hours * 3600 * (sample_rate / hop_length))

    n_items = len(dataset)
    if max_items is not None:
        n_items = min(n_items, max_items)

    for idx in range(n_items):
        ex = dataset[idx]
        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
        if wav.dim() == 2:
            wav = wav.mean(0)
        wav = wav.unsqueeze(0)
        mel = mel_fn(wav)
        logmel = torch.log(mel.clamp_min(1e-10))
        logmel = logmel.squeeze(0)
        if max_frames is not None and logmel.size(1) > max_frames:
            logmel = logmel[:, :max_frames]
        sum_f += logmel.sum(dim=1)
        sumsq_f += (logmel**2).sum(dim=1)
        count += logmel.size(1)

    mean = (sum_f / count).float()
    var = (sumsq_f / count) - (mean.double() ** 2)
    std = torch.sqrt(torch.clamp(var, min=eps)).float()

    out = {"mean": mean, "std": std}
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {"mean": mean.cpu(), "std": std.cpu()},
            save_path,
        )
    return out


DEFAULT_GLOBAL_CMVN_PATH = "global_cmvn_librispeech_clean.pt"


def load_librispeech(
    sp_model_path: str,
    batch_size=32,
    eval_batch_size=None,
    num_workers=8,
    eval_num_workers=0,
    val_max_size=None,
    seed=0,
    max_train_samples=None,
    include_val_in_train=False,
    max_eval_batches=50,
    cmvn_mode: str = "utt",
    global_cmvn_path: Optional[str] = None,
    compute_global_cmvn: bool = False,
    cmvn_max_hours: Optional[float] = None,
    cmvn_eps: float = 1e-5,
    label_noise_mode: str = "none",
    label_noise_p: float = 0.0,
    label_noise_k: int = 0,
    label_noise_seed: int = 0,
    label_noise_replace: str = "random",
    label_noise_disk_path: Optional[str] = None,
    label_noise_use_disk: bool = False,
    label_noise_save_disk: bool = False,
    rt_noise_k: int = 0,
    rt_noise_p: float = 0.0,
    rt_noise_seed: Optional[int] = None,
    rt_noise_replace: str = "random",
):
    """
    Load LibriSpeech (openslr/librispeech_asr clean), encode with SentencePiece.
    - eval_batch_size: batch size for val/test/eval loaders (default: same as batch_size).
    - num_workers: workers for train_dl only.
    - eval_num_workers: workers for val/test/train_eval loaders (default 0 to save RAM).
    - max_train_samples: cap training set size for quick experiments (default: use all).
    - include_val_in_train: if True, train on train+val (validation not used for tuning).
    - max_eval_batches: fixed number of batches for train_eval_dl (same batches every run, same seed).
    - cmvn_mode: "none" | "utt" | "global" | "batch". Default "utt" (per-utterance CMVN).
    - global_cmvn_path: path to load/save global CMVN stats when cmvn_mode=="global".
      If None, uses DEFAULT_GLOBAL_CMVN_PATH so stats are reused across runs.
    - compute_global_cmvn: if True and path missing, compute global stats from train and save.
    - cmvn_max_hours: optional cap (hours of audio) when computing global CMVN for speed.
    - cmvn_eps: epsilon for std in normalization.
    Returns: train_dl, val_dl, test_dl, train_eval_dl, sp, blank_id.
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    blank_id = sp.piece_to_id("<blank>")
    assert blank_id >= 0, "SentencePiece model must contain <blank> token"

    def _encode(ex):
        return encode_example(ex, sp)

    train_ds1 = load_dataset("openslr/librispeech_asr", "clean", split="train.360")
    train_ds2 = load_dataset("openslr/librispeech_asr", "clean", split="train.100")
    train_ds = concatenate_datasets([train_ds1, train_ds2])
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    train_ds = train_ds.shuffle(seed=seed)
    train_ds = train_ds.map(_encode, load_from_cache_file=True)

    val_ds = load_dataset("openslr/librispeech_asr", "clean", split="validation")
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=16000))
    if val_max_size is not None:
        val_ds = val_ds.select(range(min(val_max_size, len(val_ds))))
    val_ds = val_ds.map(_encode, load_from_cache_file=True)

    if include_val_in_train:
        train_ds = concatenate_datasets([train_ds, val_ds])
        train_ds = train_ds.shuffle(seed=seed + 1)

    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))

    vocab_size = sp.get_piece_size()
    try:
        unk_id = sp.piece_to_id("<unk>")
    except Exception:
        unk_id = 0
    if unk_id < 0:
        unk_id = 0
    if label_noise_replace == "unk" and (unk_id < 0 or unk_id >= vocab_size):
        raise ValueError("SP model has no <unk>; cannot use label_noise_replace='unk'")

    if label_noise_mode in ("static", "static_disk"):
        n_train = len(train_ds)
        sample_idx = [0, min(1, n_train - 1), min(2, n_train - 1)]
        if n_train < 3:
            sample_idx = list(range(n_train))
        sample_clean = [train_ds[i]["targets"] for i in sample_idx]

        use_disk = label_noise_use_disk or (label_noise_mode == "static_disk")
        save_disk = label_noise_save_disk or (label_noise_mode == "static_disk")
        path = label_noise_disk_path or "static_label_noise.pt"
        fingerprint = getattr(train_ds, "_fingerprint", None) or str(len(train_ds))
        corrupted = load_static_noise_cache(
            path if use_disk else "",
            len(train_ds),
            fingerprint,
            label_noise_seed,
            label_noise_p,
            label_noise_k,
            label_noise_replace,
            vocab_size,
        )
        if corrupted is None:

            def _corrupt_fn(ex, idx):
                t = ex["targets"]
                c = _corrupt_targets_static(
                    t,
                    idx,
                    label_noise_seed,
                    label_noise_p,
                    label_noise_k,
                    label_noise_replace,
                    vocab_size,
                    blank_id,
                    unk_id,
                )
                return {"targets": c}

            train_ds = train_ds.map(_corrupt_fn, with_indices=True)
            if save_disk:
                corrupted_save = [train_ds[i]["targets"] for i in range(len(train_ds))]
                save_static_noise_cache(
                    path,
                    corrupted_save,
                    fingerprint,
                    label_noise_seed,
                    label_noise_p,
                    label_noise_k,
                    label_noise_replace,
                    vocab_size,
                )
        else:
            train_ds = train_ds.remove_columns("targets").add_column("targets", corrupted)

        sample_noisy = [train_ds[i]["targets"] for i in sample_idx]
        for i, (clean, noisy) in enumerate(zip(sample_clean, sample_noisy)):
            clean_txt = sp.decode(clean) if clean else ""
            noisy_txt = sp.decode(noisy) if noisy else ""
            print(
                f"[label_noise] sample {i} before: {clean_txt[:80]!r} ... "
                f"after: {noisy_txt[:80]!r} ..."
            )
        assert blank_id not in [
            t for seq in sample_noisy for t in seq
        ], "blank_id must not appear in corrupted targets"

    test_ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
    test_ds = test_ds.map(_encode)

    global_stats = None
    if cmvn_mode == "global":
        path = global_cmvn_path or DEFAULT_GLOBAL_CMVN_PATH
        if not compute_global_cmvn and os.path.isfile(path):
            d = torch.load(path, map_location="cpu", weights_only=True)
            global_stats = {"mean": d["mean"], "std": d["std"]}
        else:
            global_stats = compute_global_cmvn_stats(
                train_ds,
                sample_rate=16000,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                win_length=400,
                max_hours=cmvn_max_hours,
                eps=cmvn_eps,
                save_path=path,
            )

    collate_fn = CollateCTC(
        pad_id=blank_id,
        cmvn_mode=cmvn_mode,
        global_stats=global_stats,
        eps=cmvn_eps,
        rt_noise_k=rt_noise_k,
        rt_noise_p=rt_noise_p,
        rt_noise_seed=rt_noise_seed,
        rt_noise_replace=rt_noise_replace,
        vocab_size=vocab_size,
        unk_id=unk_id,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=eval_num_workers,
        pin_memory=(eval_num_workers > 0),
        persistent_workers=(eval_num_workers > 0),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=eval_num_workers,
        pin_memory=(eval_num_workers > 0),
        persistent_workers=(eval_num_workers > 0),
    )
    n_train_eval = min(max_eval_batches * eval_batch_size, len(train_ds))
    train_eval_ds = train_ds.select(range(n_train_eval))
    train_eval_dl = DataLoader(
        train_eval_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=eval_num_workers,
        pin_memory=(eval_num_workers > 0),
        persistent_workers=(eval_num_workers > 0),
    )
    return train_dl, val_dl, test_dl, train_eval_dl, sp, blank_id

#!/usr/bin/env python3
"""Train a SentencePiece tokenizer for CTC (blank=0, unk=1)."""
import argparse
import os
import tempfile

import sentencepiece as spm

from dataset import normalize_text


def train_from_librispeech(
    model_prefix: str,
    vocab_size: int = 1000,
    character_coverage: float = 1.0,
):
    """Load LibriSpeech train text, normalize, write to temp file, train SPM."""
    from datasets import load_dataset, Audio, concatenate_datasets

    train_ds1 = load_dataset(
        "openslr/librispeech_asr", "clean", split="train.360"
    )
    train_ds2 = load_dataset(
        "openslr/librispeech_asr", "clean", split="train.100"
    )
    train_ds = concatenate_datasets([train_ds1, train_ds2])
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    texts = [normalize_text(x["text"]) for x in train_ds]
    texts = [t for t in texts if t]
    return train_from_texts(
        texts, model_prefix, vocab_size, character_coverage
    )


def train_from_text_file(
    text_path: str,
    model_prefix: str,
    vocab_size: int = 1000,
    character_coverage: float = 1.0,
):
    """Train SPM from a text file (one sentence per line)."""
    with open(text_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return train_from_texts(
        lines, model_prefix, vocab_size, character_coverage
    )


def train_from_texts(
    texts: list[str],
    model_prefix: str,
    vocab_size: int = 1000,
    character_coverage: float = 1.0,
):
    """Train SentencePiece unigram model. Uses CTC convention: pad_id=0 (<blank>), unk_id=1."""
    if not texts:
        raise ValueError("No texts to train on.")
    fd, input_path = tempfile.mkstemp(suffix=".txt", prefix="spm_train_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t + "\n")
        spm.SentencePieceTrainer.Train(
            input=input_path,
            model_prefix=model_prefix,
            model_type="unigram",
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            pad_id=0,
            pad_piece="<blank>",
            unk_id=1,
            unk_piece="<unk>",
            bos_id=-1,
            eos_id=-1,
        )
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass
    return model_prefix + ".model"


def main():
    p = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer (CTC: blank=0, unk=1)"
    )
    p.add_argument(
        "--source",
        choices=("librispeech", "file"),
        default="librispeech",
        help="Data source: librispeech (train.360+100) or file",
    )
    p.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to text file (one sentence per line). Required if --source file",
    )
    p.add_argument(
        "--output",
        "--model_prefix",
        dest="model_prefix",
        type=str,
        default="spm_unigram",
        help="Output model prefix (e.g. spm_unigram -> spm_unigram.model + .vocab)",
    )
    p.add_argument(
        "--vocab_size", type=int, default=1000, help="Vocabulary size"
    )
    p.add_argument(
        "--character_coverage",
        type=float,
        default=1.0,
        help="Character coverage (1.0 for English)",
    )
    args = p.parse_args()

    if args.source == "file":
        if not args.text_file or not os.path.isfile(args.text_file):
            p.error(
                "--source file requires --text_file pointing to an existing file"
            )
        train_from_text_file(
            args.text_file,
            args.model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
        )
    else:
        train_from_librispeech(
            args.model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
        )

    model_path = args.model_prefix + ".model"
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    blank_id = sp.piece_to_id("<blank>")
    unk_id = sp.piece_to_id("<unk>")
    print(f"Saved: {model_path} and {args.model_prefix}.vocab")
    print(
        f"blank_id={blank_id} unk_id={unk_id} vocab_size={sp.get_piece_size()}"
    )
    assert blank_id == 0, "CTC expects blank_id=0"


if __name__ == "__main__":
    main()

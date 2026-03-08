import re
import sentencepiece as spm

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
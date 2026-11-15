"""Arabic text normalization and tokenization for lexical retrieval."""
from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat, Quranic marks, tatweel
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)

_CHAR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
})

# Light stemming: ordered longest-first so the most specific affix wins.
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و")
_SUFFIXES = ("هما", "كما", "ات", "ان", "ون", "ين", "يه", "ها", "هم", "ه", "ي")
_MIN_STEM = 3

# Short, high-frequency function words. Kept small on purpose: question words
# (من، ما، متى، أين، كم) are dropped because they carry no topical signal.
STOPWORDS = frozenset(
    "في من الي علي عن ان او ثم هو هي هم ما ماذا متي اين كيف كم لماذا هل التي الذي الذين "
    "هذا هذه ذلك تلك كان كانت مع قد لا لم لن بين كل بعد قبل عند حتي اي".split()
)


def normalize(text: str) -> str:
    """Unify letter variants, strip diacritics/tatweel and punctuation, lowercase Latin."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_CHAR_MAP).lower()
    return _NON_WORD.sub(" ", text).strip()


def light_stem(token: str) -> str:
    """Strip at most one common prefix and one common suffix, keeping >= 3 letters."""
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= _MIN_STEM:
            token = token[len(p):]
            break
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= _MIN_STEM:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    tokens = normalize(text).split()
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return tokens

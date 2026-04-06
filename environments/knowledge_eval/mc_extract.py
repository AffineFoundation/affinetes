"""Multiple-choice answer extractor for arbitrary label sets.

Accepts an explicit list of valid labels (letters, numbers, or
mixed) and walks several heuristics from strict to loose. We never
guess outside the provided label set.
"""

import re
from typing import Iterable, List, Optional


_THINK_TAGS = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _normalize(label: str) -> str:
    return label.strip().upper()


def _build_alternation(labels: Iterable[str]) -> str:
    """Build a regex alternation that matches the longest label first
    (so ``10`` is preferred over ``1`` when both are in the set).
    """
    sorted_labels = sorted({_normalize(l) for l in labels}, key=lambda s: (-len(s), s))
    return "|".join(re.escape(l) for l in sorted_labels)


def extract_choice(response: str, valid_labels: Iterable[str]) -> Optional[str]:
    """Extract one of ``valid_labels`` from ``response``.

    Returns the matched label (in the same case as it appeared in
    ``valid_labels`` after normalisation), or ``None`` if nothing
    matches.
    """
    if not response:
        return None
    valid = {_normalize(l) for l in valid_labels}
    if not valid:
        return None

    text = _THINK_TAGS.sub("", response).strip()
    if not text:
        return None

    alt = _build_alternation(valid)

    # 1) "Answer: X" / "Final answer: X" — strongest signal, scan from end.
    #    The (?!\w) lookahead prevents matching a numeric label as a
    #    prefix of a longer number (e.g. don't return '3' from 'Answer: 31'
    #    when 31 is outside the valid set).
    pat = re.compile(
        rf"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\(?\s*({alt})(?!\w)\s*\)?",
        re.IGNORECASE,
    )
    matches = list(pat.finditer(text))
    for m in reversed(matches):
        cand = _normalize(m.group(1))
        if cand in valid:
            return cand

    # 2) \boxed{X}
    pat = re.compile(rf"\\boxed\{{\s*\(?\s*({alt})\s*\)?\s*\}}", re.IGNORECASE)
    matches = list(pat.finditer(text))
    for m in reversed(matches):
        cand = _normalize(m.group(1))
        if cand in valid:
            return cand

    # 3) Last (X) parenthesised
    pat = re.compile(rf"\(\s*({alt})\s*\)")
    matches = list(pat.finditer(text))
    for m in reversed(matches):
        cand = _normalize(m.group(1))
        if cand in valid:
            return cand

    # 4) Bare label in the tail of the response (avoid grabbing labels
    #    from the question restatement at the top).
    tail = text[-200:]
    pat = re.compile(rf"\b({alt})\b")
    matches = list(pat.finditer(tail))
    for m in reversed(matches):
        cand = _normalize(m.group(1))
        if cand in valid:
            return cand

    return None

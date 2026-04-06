"""Shared multiple-choice answer extraction utilities.

The extractor accepts a model response and the set of valid letter labels
(e.g. {"A","B","C","D"} for GPQA, {"A".."J"} for MMLU-Pro). It first strips
``<think>`` style scratchpads, then walks several heuristics from strict to
loose so that we don't reward sloppy outputs but still tolerate common
formatting drift.
"""

import re
from typing import Optional, Set

_THINK_TAGS = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\(?\s*([A-J])\s*\)?",
    re.IGNORECASE,
)
_BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-J])\s*\)?\s*\}", re.IGNORECASE)
_PAREN = re.compile(r"\(\s*([A-J])\s*\)")
_BARE = re.compile(r"\b([A-J])\b")


def extract_choice(response: str, valid_letters: Set[str]) -> Optional[str]:
    """Extract a single uppercase letter from ``response``.

    Returns ``None`` if no candidate falls inside ``valid_letters``.
    """
    if not response:
        return None

    text = _THINK_TAGS.sub("", response).strip()
    if not text:
        return None

    valid = {c.upper() for c in valid_letters}

    # 1. "Answer: X" / "Final answer: X" — strongest signal, scan from the end
    for match in reversed(list(_ANSWER_LINE.finditer(text))):
        letter = match.group(1).upper()
        if letter in valid:
            return letter

    # 2. \boxed{X}
    for match in reversed(list(_BOXED.finditer(text))):
        letter = match.group(1).upper()
        if letter in valid:
            return letter

    # 3. Last (X) parenthesised letter
    for match in reversed(list(_PAREN.finditer(text))):
        letter = match.group(1).upper()
        if letter in valid:
            return letter

    # 4. Last bare uppercase letter on its own — only consider the tail of the
    #    response so we don't grab letters from the question restatement.
    tail = text[-200:]
    for match in reversed(list(_BARE.finditer(tail))):
        letter = match.group(1).upper()
        if letter in valid:
            return letter

    return None

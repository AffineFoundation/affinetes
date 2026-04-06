"""Helpers for batched (multi-question, single-call) evaluation.

The Actor packs N task prompts into a single LLM call separated by
``=== Answer K ===`` markers. These functions own the prompt template
and the response splitter so they can be unit-tested without spinning
up the full environment.
"""

import re
from typing import List

# Marker pattern is intentionally specific to avoid colliding with
# anything a model might emit naturally.
MARKER_RE = re.compile(r"===\s*Answer\s*(\d+)\s*===", re.IGNORECASE)


def build_batch_prompt(per_item_prompts: List[str]) -> str:
    """Wrap N independent prompts into one joint prompt."""
    n = len(per_item_prompts)
    header = (
        f"You will answer {n} independent questions in a single response. "
        f"Each question may come from a different benchmark and have a "
        f"different output format. Answer **each question completely** as "
        f"if it were the only question, and use the exact separator "
        f"`=== Answer K ===` (where K is 1..{n}) to delimit your "
        f"answers. Output nothing else outside the answer blocks.\n\n"
        f"Format your response exactly like this:\n\n"
    )
    for i in range(n):
        header += f"=== Answer {i + 1} ===\n<your full answer for question {i + 1}>\n\n"
    body = ""
    for i, p in enumerate(per_item_prompts):
        body += f"\n--- Question {i + 1} ---\n{p}\n"
    return header + body


def split_batch_response(response: str, n: int) -> List[str]:
    """Slice a batched response into ``n`` per-question chunks.

    Robust to:
        * out-of-order markers (we trust the marker number, not order)
        * missing markers (the corresponding chunk becomes empty)
        * extra text before the first marker (it is dropped)
        * non-contiguous numbering (only valid 1..n indices are kept)
    """
    if not response:
        return [""] * n
    matches = list(MARKER_RE.finditer(response))
    if not matches:
        # Fall back: hand the whole text to question 1 so its extractor
        # at least has something to chew on. Other slots stay empty.
        out = [""] * n
        out[0] = response.strip()
        return out

    matches_sorted = sorted(matches, key=lambda m: m.start())
    slots: List[str] = [""] * n
    for i, m in enumerate(matches_sorted):
        idx = int(m.group(1)) - 1
        if not (0 <= idx < n):
            continue
        start = m.end()
        end = matches_sorted[i + 1].start() if i + 1 < len(matches_sorted) else len(response)
        slots[idx] = response[start:end].strip()
    return slots

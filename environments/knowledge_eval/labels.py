"""Dynamic label generator for multiple-choice options.

Tasks used to hard-code A..Z (26 max). With cross_pool extension some
HLE rows already start at 26 options, so we need a label scheme that
scales arbitrarily.

Strategy:
    * n <= 26  → A..Z (preserves the convention everyone expects)
    * n  > 26  → numeric "1".."N" (clean, unambiguous, unbounded)

The two schemes are never mixed within one prompt — once we cross 26
we go full numeric for that question.
"""

from typing import List

_LETTERS = [chr(ord("A") + i) for i in range(26)]


def make_labels(n: int) -> List[str]:
    """Return a list of ``n`` distinct labels.

    Letters are used while they fit (n ≤ 26); above that we switch to
    numeric labels starting at 1.
    """
    if n <= 0:
        return []
    if n <= 26:
        return _LETTERS[:n]
    return [str(i + 1) for i in range(n)]

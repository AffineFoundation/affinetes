"""Cross-question distractor pool.

For ``anti_contam == 'distractor_swap'`` we replace a question's
incorrect options with options drawn from *other* questions in the
same broad category. This breaks any memorisation strategy that
relies on knowing the original option set, while keeping the surface
form plausible (e.g. a Physics question gets Physics-flavoured
distractors).

The pool is built once at module import from the in-memory rows.
Sampling is keyed on a deterministic ``random.Random`` so the same
``(task_id, perturb_seed)`` always yields the same swapped options.
"""

import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# Meta options like "None of the above" / "All of the above" are
# meaningless when transplanted to a different question, so we drop
# them at pool-build time. The pattern is intentionally narrow — we
# only want to skip phrases whose meaning depends on the *current*
# question's option list, not generic short phrases.
_META_OPTION_RE = re.compile(
    r"\b("
    r"none\s+of\s+the\s+(above|other|preceding|following)"
    r"|all\s+of\s+the\s+(above|other|preceding|following)"
    r"|both\s+[a-z]\s+and\s+[a-z]"
    r"|^[a-z]\s+and\s+[a-z]$"
    r"|all\s+answers?\s+are\s+correct"
    r"|none\s+of\s+the\s+(other\s+)?answers?\s+(are|is)\s+correct"
    r")",
    re.IGNORECASE,
)


def _is_meta_option(text: str) -> bool:
    return bool(_META_OPTION_RE.search(text or ""))


class DistractorPool:
    """Per-category pools of (source_index, option_text) tuples.

    Two pools live side by side:

    * ``_distractors_by_category`` — only the *incorrect* options.
      Used by ``shuffle`` / ``distractor_swap`` modes.
    * ``_all_options_by_category`` — distractors **and** correct
      answers from every other question in the category. Used by
      ``cross_pool`` so the model is forced to pick the right answer
      out of a soup that also contains *other questions' correct
      answers*. This breaks any "I recognise this as a textbook
      correct answer" heuristic.
    """

    def __init__(self, rows: List[Dict[str, Any]], task_type: str) -> None:
        self.task_type = task_type
        self._distractors_by_category: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self._all_options_by_category: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self._all_distractors: List[Tuple[int, str]] = []
        self._all_options: List[Tuple[int, str]] = []

        for idx, row in enumerate(rows):
            cat = (row.get("category") or "").strip() or "Unknown"
            distractors, correct = self._extract_options(row)
            for text in distractors:
                cleaned = (text or "").strip()
                if not cleaned or _is_meta_option(cleaned):
                    continue
                self._distractors_by_category[cat].append((idx, cleaned))
                self._all_distractors.append((idx, cleaned))
                self._all_options_by_category[cat].append((idx, cleaned))
                self._all_options.append((idx, cleaned))
            if correct and not _is_meta_option(correct):
                self._all_options_by_category[cat].append((idx, correct))
                self._all_options.append((idx, correct))

    @staticmethod
    def _extract_options(row: Dict[str, Any]) -> Tuple[List[str], str]:
        """Return ``(distractors, correct_text)`` for a row.

        For GPQA we have explicit ``correct`` + ``distractors``. For
        MMLU-Pro we have ``options`` + an ``answer`` letter, so we
        split on the gold index.
        """
        if "distractors" in row:
            correct = (row.get("correct") or "").strip()
            distractors = [str(d).strip() for d in (row.get("distractors") or [])]
            return distractors, correct
        options = [str(o).strip() for o in (row.get("options") or [])]
        answer = (row.get("answer") or "").strip().upper()
        if not options or len(answer) != 1:
            return [], ""
        gold_idx = ord(answer) - ord("A")
        if not (0 <= gold_idx < len(options)):
            return [], ""
        correct = options[gold_idx]
        distractors = [opt for i, opt in enumerate(options) if i != gold_idx]
        return distractors, correct

    def _sample_from(
        self,
        primary_pool: List[Tuple[int, str]],
        backup_pool: List[Tuple[int, str]],
        exclude_idx: int,
        forbidden_texts: List[str],
        n: int,
        rng: random.Random,
    ) -> List[str]:
        forbidden = {_norm(t) for t in forbidden_texts if t}
        seen: set[str] = set(forbidden)
        result: List[str] = []

        for pool in (primary_pool, backup_pool):
            candidates = [
                (i, t) for (i, t) in pool
                if i != exclude_idx and _norm(t) not in seen
            ]
            rng.shuffle(candidates)
            for _, text in candidates:
                key = _norm(text)
                if key in seen:
                    continue
                seen.add(key)
                result.append(text)
                if len(result) == n:
                    return result
        return result

    def sample(
        self,
        category: str,
        exclude_idx: int,
        forbidden_texts: List[str],
        n: int,
        rng: random.Random,
    ) -> List[str]:
        """Pick ``n`` *incorrect* distractors not from question ``exclude_idx``.

        Used by ``distractor_swap`` mode. Falls back to the global
        distractor pool when the same-category pool is too thin.
        """
        return self._sample_from(
            primary_pool=self._distractors_by_category.get(category, []),
            backup_pool=self._all_distractors,
            exclude_idx=exclude_idx,
            forbidden_texts=forbidden_texts,
            n=n,
            rng=rng,
        )

    def sample_cross(
        self,
        category: str,
        exclude_idx: int,
        forbidden_texts: List[str],
        n: int,
        rng: random.Random,
    ) -> List[str]:
        """Pick ``n`` options (distractors **or** other-question correct
        answers) not from question ``exclude_idx``.

        Used by ``cross_pool`` mode. The point is to mix other
        questions' *correct* answers into the distractor set, so a
        memorised "this looks like a known correct answer" heuristic
        becomes useless: it's a correct answer, just not for *this*
        question.
        """
        return self._sample_from(
            primary_pool=self._all_options_by_category.get(category, []),
            backup_pool=self._all_options,
            exclude_idx=exclude_idx,
            forbidden_texts=forbidden_texts,
            n=n,
            rng=rng,
        )

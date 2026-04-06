"""IFEval task with a hand-rolled subset of instruction verifiers.

We deliberately implement only a small list of constraints that can be
checked with the standard library, and the preprocessor drops any IFEval
sample whose instructions reach outside this set. This keeps the docker
image lean (no nltk/langdetect) while still exercising real instruction
following behaviour. Strict scoring is used: a sample scores 1.0 only
when *every* listed instruction passes.
"""

import re
from typing import Any, Dict, List, Tuple

from models import Challenge

# ---- Supported instruction ids -------------------------------------------------
# Names match the upstream IFEval ``instruction_id`` registry exactly so the
# preprocessor can filter on them. Each id maps to a verifier function below.

SUPPORTED_INSTRUCTIONS = {
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "startend:end_checker",
    "startend:quotation",
    "punctuation:no_comma",
    "detectable_format:number_bullet_lists",
}


# ---- Text helpers --------------------------------------------------------------

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    # Naive splitter — IFEval's official verifier uses nltk; this is good enough
    # for the simple constraints we keep, and avoids a 30MB dependency.
    parts = _SENT_SPLIT.split(text)
    return sum(1 for p in parts if p.strip())


def _count_paragraphs(text: str) -> int:
    parts = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(parts)


def _comparison_ok(actual: int, relation: str, target: int) -> bool:
    relation = (relation or "").lower()
    if relation in ("at least", "atleast", "at_least", "minimum", "min"):
        return actual >= target
    if relation in ("at most", "atmost", "at_most", "maximum", "max", "less than"):
        return actual <= target
    if relation in ("less than", "fewer than"):
        return actual < target
    if relation in ("more than", "greater than"):
        return actual > target
    return actual == target


# ---- Individual verifiers ------------------------------------------------------

def _v_keywords_existence(text: str, kw: Dict[str, Any]) -> bool:
    keywords = kw.get("keywords") or []
    lowered = text.lower()
    return all(str(k).lower() in lowered for k in keywords)


def _v_keywords_forbidden(text: str, kw: Dict[str, Any]) -> bool:
    forbidden = kw.get("forbidden_words") or kw.get("keywords") or []
    lowered = text.lower()
    return all(str(k).lower() not in lowered for k in forbidden)


def _v_keywords_frequency(text: str, kw: Dict[str, Any]) -> bool:
    keyword = kw.get("keyword")
    if not keyword:
        return True
    target = int(kw.get("frequency", 0))
    relation = kw.get("relation", "at least")
    actual = len(re.findall(rf"\b{re.escape(str(keyword))}\b", text, re.IGNORECASE))
    return _comparison_ok(actual, relation, target)


def _v_number_words(text: str, kw: Dict[str, Any]) -> bool:
    target = int(kw.get("num_words", 0))
    relation = kw.get("relation", "at least")
    return _comparison_ok(_count_words(text), relation, target)


def _v_number_sentences(text: str, kw: Dict[str, Any]) -> bool:
    target = int(kw.get("num_sentences", 0))
    relation = kw.get("relation", "at least")
    return _comparison_ok(_count_sentences(text), relation, target)


def _v_number_paragraphs(text: str, kw: Dict[str, Any]) -> bool:
    target = int(kw.get("num_paragraphs", 0))
    return _count_paragraphs(text) == target


def _v_lowercase(text: str, kw: Dict[str, Any]) -> bool:
    return text == text.lower()


def _v_uppercase(text: str, kw: Dict[str, Any]) -> bool:
    return text == text.upper()


def _v_end_checker(text: str, kw: Dict[str, Any]) -> bool:
    suffix = kw.get("end_phrase") or ""
    return text.rstrip().endswith(str(suffix))


def _v_quotation(text: str, kw: Dict[str, Any]) -> bool:
    stripped = text.strip()
    return len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"')


def _v_no_comma(text: str, kw: Dict[str, Any]) -> bool:
    return "," not in text


def _v_number_bullet_lists(text: str, kw: Dict[str, Any]) -> bool:
    target = int(kw.get("num_bullets", 0))
    bullets = re.findall(r"^\s*(?:[-*]|\d+\.)\s+", text, re.MULTILINE)
    return len(bullets) == target


_VERIFIERS = {
    "keywords:existence": _v_keywords_existence,
    "keywords:forbidden_words": _v_keywords_forbidden,
    "keywords:frequency": _v_keywords_frequency,
    "length_constraints:number_words": _v_number_words,
    "length_constraints:number_sentences": _v_number_sentences,
    "length_constraints:number_paragraphs": _v_number_paragraphs,
    "change_case:english_lowercase": _v_lowercase,
    "change_case:english_capital": _v_uppercase,
    "startend:end_checker": _v_end_checker,
    "startend:quotation": _v_quotation,
    "punctuation:no_comma": _v_no_comma,
    "detectable_format:number_bullet_lists": _v_number_bullet_lists,
}


# ---- Task interface ------------------------------------------------------------

class IFEvalTask:
    """Strict-mode IFEval task. The full prompt is sent verbatim.

    The ``mode`` / ``perturb_seed`` / ``pool`` kwargs are accepted for
    interface parity with the multiple-choice tasks but ignored — there
    is nothing to shuffle in an IFEval prompt.
    """

    def __init__(self) -> None:
        pass

    async def generate(
        self,
        sample: Dict[str, Any],
        task_id: int,
        *,
        mode: str = "shuffle",
        perturb_seed: int = 0,
        pool: Any = None,
        extra_distractors: int = 0,  # accepted for interface parity, unused
    ) -> Challenge:
        return Challenge(
            env="knowledge_eval:ifeval",
            prompt=sample["prompt"],
            extra={
                "instruction_id_list": sample.get("instruction_id_list", []),
                "kwargs": sample.get("kwargs", []),
                "task_id": task_id,
                "mode": "off",  # always reported as "off" since no perturbation applies
                "perturb_seed": perturb_seed,
            },
        )

    async def evaluate(self, response: str, challenge: Challenge) -> float:
        if not response:
            return 0.0
        instructions: List[str] = challenge.extra.get("instruction_id_list", [])
        kwargs_list: List[Dict[str, Any]] = challenge.extra.get("kwargs", [])
        if not instructions:
            return 0.0

        for idx, iid in enumerate(instructions):
            verifier = _VERIFIERS.get(iid)
            if verifier is None:
                # Unknown id slipped through preprocessing — fail closed.
                return 0.0
            kw = kwargs_list[idx] if idx < len(kwargs_list) else {}
            try:
                ok = verifier(response, kw or {})
            except Exception:
                ok = False
            if not ok:
                return 0.0
        return 1.0

    def detail(self, response: str, challenge: Challenge) -> Tuple[float, List[Dict[str, Any]]]:
        """Return per-instruction breakdown for debugging / dumps."""
        instructions: List[str] = challenge.extra.get("instruction_id_list", [])
        kwargs_list: List[Dict[str, Any]] = challenge.extra.get("kwargs", [])
        breakdown: List[Dict[str, Any]] = []
        all_ok = True
        for idx, iid in enumerate(instructions):
            verifier = _VERIFIERS.get(iid)
            kw = kwargs_list[idx] if idx < len(kwargs_list) else {}
            if verifier is None:
                ok = False
            else:
                try:
                    ok = verifier(response or "", kw or {})
                except Exception:
                    ok = False
            all_ok = all_ok and ok
            breakdown.append({"instruction_id": iid, "passed": ok, "kwargs": kw})
        return (1.0 if all_ok and instructions else 0.0), breakdown

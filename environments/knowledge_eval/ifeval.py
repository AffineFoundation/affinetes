"""IFEval task with a hand-rolled subset of instruction verifiers.

We deliberately implement only a small list of constraints that can be
checked with the standard library, and the preprocessor drops any IFEval
sample whose instructions reach outside this set. This keeps the docker
image lean (no nltk/langdetect) while still exercising real instruction
following behaviour. Strict scoring is used: a sample scores 1.0 only
when *every* listed instruction passes.

Variants
========

When ``perturb_seed > 0`` we synthesise a fresh prompt instead of
sending the upstream IFEval text:

  * the *base task* is drawn from ``ifeval_base_prompts.BASE_PROMPTS``
    (a curated pool of open-ended generative tasks)
  * each instruction's kwargs are deterministically re-rolled from
    realistic pools (collected at preprocess time from every IFEval
    row, see ``ifeval_pools.json``)
  * an instruction → natural language renderer expresses the new
    constraints in plain English so the model can follow them

The verifier path doesn't change at all — it operates on the model
response and the (possibly perturbed) kwargs stored in
``challenge.extra["kwargs"]``.

When ``perturb_seed == 0`` we keep the original behaviour for
backward compatibility, so existing rollouts and the canonical layer
``[12743, 12977)`` remain unchanged.
"""

import json
import random
import re
from pathlib import Path
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


# ---- Variant generation: pools, kwargs perturbation, instruction rendering ----

_POOLS_PATH = Path("/app/data/ifeval_pools.json")


def _load_pools() -> Dict[str, List[str]]:
    """Load keyword pools collected by preprocess. Returns empty pools
    when the file is missing so unit tests can run outside the container.
    """
    if not _POOLS_PATH.exists():
        return {"existence_keywords": [], "forbidden_words": [], "frequency_keywords": []}
    try:
        return json.loads(_POOLS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"existence_keywords": [], "forbidden_words": [], "frequency_keywords": []}


_POOLS = _load_pools()

# ``BASE_PROMPTS`` is imported lazily because the unit tests in /tmp may
# import this module without /app on sys.path.
try:
    from ifeval_base_prompts import BASE_PROMPTS as _BASE_PROMPTS
except Exception:
    _BASE_PROMPTS = ["Write a short essay on a topic of your choice."]


def _perturb_kwargs(rng: random.Random, iid: str, original: Dict[str, Any]) -> Dict[str, Any]:
    """Re-roll an instruction's kwargs from realistic pools.

    Instructions with no tunable kwargs (``no_comma``, lowercase, etc.)
    are returned untouched.
    """
    if iid == "keywords:existence":
        n = max(1, len(original.get("keywords") or [3]) or 3)
        pool = _POOLS["existence_keywords"] or [
            "harmony", "discovery", "courage", "wisdom", "wonder",
        ]
        n = min(n, len(pool))
        return {"keywords": rng.sample(pool, k=n)}

    if iid == "keywords:forbidden_words":
        n = max(1, len(original.get("forbidden_words") or [3]) or 3)
        pool = _POOLS["forbidden_words"] or [
            "however", "moreover", "actually", "basically", "literally",
        ]
        n = min(n, len(pool))
        return {"forbidden_words": rng.sample(pool, k=n)}

    if iid == "keywords:frequency":
        pool = _POOLS["frequency_keywords"] or ["wonder", "harmony", "balance", "vision"]
        return {
            "keyword": rng.choice(pool),
            "frequency": rng.randint(2, 10),
            "relation": "at least",
        }

    if iid == "length_constraints:number_words":
        return {
            "num_words": rng.randint(50, 300),
            "relation": rng.choice(["at least", "at most"]),
        }

    if iid == "length_constraints:number_sentences":
        return {
            "num_sentences": rng.randint(3, 15),
            "relation": rng.choice(["at least", "at most"]),
        }

    if iid == "length_constraints:number_paragraphs":
        return {"num_paragraphs": rng.randint(2, 6)}

    if iid == "detectable_format:number_bullet_lists":
        return {"num_bullets": rng.randint(3, 8)}

    if iid == "startend:end_checker":
        suffix = rng.choice([
            "Thank you for reading.",
            "That is all.",
            "End of response.",
            "I hope this helps.",
        ])
        return {"end_phrase": suffix}

    # No tunable kwargs — preserve as-is.
    return dict(original or {})


def _render_instruction(iid: str, kw: Dict[str, Any]) -> str:
    """Express a (possibly perturbed) instruction in natural language so
    the model has something to follow. Phrasings deliberately mirror
    the upstream IFEval style."""
    if iid == "keywords:existence":
        kws = kw.get("keywords") or []
        return f"Include the following keywords in your response: {', '.join(kws)}."
    if iid == "keywords:forbidden_words":
        kws = kw.get("forbidden_words") or []
        return f"Do not use any of the following words: {', '.join(kws)}."
    if iid == "keywords:frequency":
        return (
            f"The word \"{kw.get('keyword', '')}\" must appear "
            f"{kw.get('relation', 'at least')} {kw.get('frequency', 1)} times."
        )
    if iid == "length_constraints:number_words":
        return (
            f"Your response must contain {kw.get('relation', 'at least')} "
            f"{kw.get('num_words', 100)} words."
        )
    if iid == "length_constraints:number_sentences":
        return (
            f"Your response must contain {kw.get('relation', 'at least')} "
            f"{kw.get('num_sentences', 5)} sentences."
        )
    if iid == "length_constraints:number_paragraphs":
        return (
            f"Your response must contain exactly "
            f"{kw.get('num_paragraphs', 3)} paragraphs, separated by a blank line."
        )
    if iid == "change_case:english_lowercase":
        return "Write your entire response in lowercase letters only."
    if iid == "change_case:english_capital":
        return "Write your entire response in UPPERCASE letters only."
    if iid == "startend:end_checker":
        return f"End your response with the exact phrase: {kw.get('end_phrase', '')}"
    if iid == "startend:quotation":
        return "Wrap your entire response in double quotation marks."
    if iid == "punctuation:no_comma":
        return "Do not use any commas anywhere in your response."
    if iid == "detectable_format:number_bullet_lists":
        return (
            f"Your response must contain exactly {kw.get('num_bullets', 3)} "
            f"bullet points (lines starting with '- ' or '1.')."
        )
    return f"Constraint: {iid}"


def _build_perturbed_prompt(
    base: str,
    instructions: List[str],
    kwargs_list: List[Dict[str, Any]],
) -> str:
    lines = [
        f"- {_render_instruction(iid, kw)}"
        for iid, kw in zip(instructions, kwargs_list)
    ]
    return (
        f"{base}\n\n"
        f"Follow these constraints in your response:\n"
        + "\n".join(lines)
    )


# ---- Task interface ------------------------------------------------------------

class IFEvalTask:
    """Strict-mode IFEval task with optional base / kwargs perturbation.

    ``perturb_seed=0`` keeps the upstream prompt + kwargs (backward
    compatible). ``perturb_seed>0`` deterministically swaps in a base
    prompt from the curated pool and re-rolls kwargs from the keyword
    pools collected at preprocess time.

    ``mode`` / ``pool`` / ``extra_distractors`` are accepted for
    interface parity with the multiple-choice tasks and ignored.
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
        extra_distractors: int = 0,
    ) -> Challenge:
        instructions: List[str] = list(sample.get("instruction_id_list") or [])
        original_kwargs: List[Dict[str, Any]] = list(sample.get("kwargs") or [])

        if perturb_seed == 0:
            prompt = sample["prompt"]
            kwargs_list = original_kwargs
            base_idx = None
        else:
            rng = random.Random(f"ifeval:{task_id}:{perturb_seed}")
            base_idx = rng.randrange(len(_BASE_PROMPTS))
            base = _BASE_PROMPTS[base_idx]
            kwargs_list = [
                _perturb_kwargs(rng, iid, kw)
                for iid, kw in zip(
                    instructions,
                    original_kwargs + [{}] * max(0, len(instructions) - len(original_kwargs)),
                )
            ]
            prompt = _build_perturbed_prompt(base, instructions, kwargs_list)

        return Challenge(
            env="knowledge_eval:ifeval",
            prompt=prompt,
            extra={
                "instruction_id_list": instructions,
                "kwargs": kwargs_list,
                "task_id": task_id,
                "mode": "off",
                "perturb_seed": perturb_seed,
                "base_prompt_idx": base_idx,
                "perturbed": perturb_seed != 0,
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

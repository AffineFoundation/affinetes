"""Build-time dataset preprocessor.

Fetches GPQA, MMLU-Pro, HLE and IFEval from HuggingFace, applies the
filtering rules described below, and writes a single ``manifest.jsonl``
plus per-task data files. This runs once inside ``docker build`` so that
the resulting image is fully self-contained and the runtime container
needs zero network access for data loading.

Filtering rules:
    GPQA       : ``Idavidrein/gpqa`` config ``gpqa_diamond`` (~198 items).
    MMLU-Pro   : ``TIGER-Lab/MMLU-Pro`` test split (~12k items).
    HLE        : ``cais/hle`` test split, only ``answer_type ==
                 'multipleChoice'`` AND no ``image`` field.
    IFEval     : ``google/IFEval`` train split, only samples whose
                 ``instruction_id_list`` is a subset of the supported
                 verifier set (see ``ifeval.SUPPORTED_INSTRUCTIONS``).

The script is intentionally tolerant: if an upstream schema field is
missing we drop the row rather than abort, so a future dataset bump
won't brick image builds.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

# Make local modules importable when invoked from /tmp during docker build.
sys.path.insert(0, str(Path(__file__).parent))

from ifeval import SUPPORTED_INSTRUCTIONS  # noqa: E402


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def load_gpqa() -> List[Dict[str, Any]]:
    """GPQA diamond — store category for cross-question distractor swap."""
    print("[preprocess] loading GPQA diamond...", flush=True)
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out: List[Dict[str, Any]] = []
    for row in ds:
        question = _safe_str(row.get("Question")).strip()
        correct = _safe_str(row.get("Correct Answer")).strip()
        incorrect = [
            _safe_str(row.get("Incorrect Answer 1")).strip(),
            _safe_str(row.get("Incorrect Answer 2")).strip(),
            _safe_str(row.get("Incorrect Answer 3")).strip(),
        ]
        if not question or not correct or not all(incorrect):
            continue
        out.append(
            {
                "question": question,
                "correct": correct,
                "distractors": incorrect,
                # "High-level domain" is the broadest GPQA category
                # (Physics / Biology / Chemistry). It's the right
                # granularity for cross-question distractor swap — too
                # narrow and the pool is empty, too broad and swapped
                # distractors look out of place.
                "category": _safe_str(row.get("High-level domain")).strip() or "Unknown",
            }
        )
    print(f"[preprocess] gpqa: {len(out)} items", flush=True)
    return out


def load_mmlu_pro() -> List[Dict[str, Any]]:
    print("[preprocess] loading MMLU-Pro...", flush=True)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out: List[Dict[str, Any]] = []
    for row in ds:
        question = _safe_str(row.get("question"))
        options = row.get("options") or []
        answer = _safe_str(row.get("answer")).strip().upper()
        if not question or not options or not answer:
            continue
        # MMLU-Pro letters go up to J (10 options). Some entries have fewer.
        if len(answer) != 1 or not ("A" <= answer <= chr(ord("A") + len(options) - 1)):
            continue
        out.append(
            {
                "question": question,
                "options": [_safe_str(o) for o in options],
                "answer": answer,
                "category": _safe_str(row.get("category")),
            }
        )
    print(f"[preprocess] mmlu_pro: {len(out)} items", flush=True)
    return out


_HLE_CHOICES_HEADER = re.compile(r"\n\s*Answer Choices?:\s*\n", re.IGNORECASE)
_HLE_OPTION_LINE = re.compile(r"^([A-Z])\.\s", re.MULTILINE)


def _parse_hle_question(question: str) -> Optional[Dict[str, Any]]:
    """Split an HLE question into ``stem`` and ``options``.

    Returns ``None`` when the structure can't be confidently recovered;
    callers should drop those rows. We require:

    * an ``Answer Choices:`` (or ``Answer Choice:``) marker line,
    * at least two option lines starting with ``X. ``,
    * the option letters form a contiguous A, B, C, ... sequence.
    """
    header = _HLE_CHOICES_HEADER.search(question)
    if header is None:
        return None
    stem = question[: header.start()].rstrip()
    body = question[header.end():]
    matches = list(_HLE_OPTION_LINE.finditer(body))
    if len(matches) < 2:
        return None
    letters = [m.group(1) for m in matches]
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if letters != expected:
        return None
    options: List[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        options.append(body[start:end].strip())
    if not stem or any(not o for o in options):
        return None
    return {"stem": stem, "options": options}


def load_hle() -> List[Dict[str, Any]]:
    """HLE multiple-choice text-only items, pre-parsed into stem+options.

    HLE samples expose ``answer_type`` (``multipleChoice`` / ``exactMatch``)
    and an ``image`` field that is an ``Image`` feature when populated.
    We disable image decoding so PIL is not required, then drop any row
    that ships an image. The remaining rows are pre-parsed so the
    runtime task module can shuffle options without re-running the
    fragile parser on every call.
    """
    from datasets import Image

    print("[preprocess] loading HLE...", flush=True)
    ds = load_dataset("cais/hle", split="test")
    for col, feat in list(ds.features.items()):
        if isinstance(feat, Image):
            ds = ds.cast_column(col, Image(decode=False))

    def _has_image(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            return bool(value.get("bytes")) or bool(value.get("path"))
        return bool(str(value).strip())

    out: List[Dict[str, Any]] = []
    parse_failures = 0
    for row in ds:
        if _safe_str(row.get("answer_type")) != "multipleChoice":
            continue
        if _has_image(row.get("image")):
            continue
        question = _safe_str(row.get("question"))
        answer = _safe_str(row.get("answer")).strip().upper()
        if not question or not answer or len(answer) != 1 or not ("A" <= answer <= "Z"):
            continue
        parsed = _parse_hle_question(question)
        if parsed is None:
            parse_failures += 1
            continue
        # Validate that the gold answer letter actually indexes an option.
        if not ("A" <= answer <= chr(ord("A") + len(parsed["options"]) - 1)):
            parse_failures += 1
            continue
        out.append(
            {
                "stem": parsed["stem"],
                "options": parsed["options"],
                "answer": answer,
                "category": _safe_str(row.get("category")),
                "id": _safe_str(row.get("id")),
            }
        )
    print(
        f"[preprocess] hle: {len(out)} items "
        f"(dropped {parse_failures} unparseable)",
        flush=True,
    )
    return out


def load_ifeval() -> List[Dict[str, Any]]:
    """Load IFEval and additionally collect keyword pools.

    The keyword pools are stitched together from **every** IFEval row
    (including ones whose instruction set we filter out), so the
    perturbed-mode samples have a rich, real-distribution pool to draw
    new keywords from. The pools are written to ``ifeval_pools.json``
    by ``main()`` after the row list is built.
    """
    print("[preprocess] loading IFEval...", flush=True)
    ds = load_dataset("google/IFEval", split="train")
    out: List[Dict[str, Any]] = []
    existence_pool: set = set()
    forbidden_pool: set = set()
    frequency_pool: set = set()
    for row in ds:
        instruction_ids = list(row.get("instruction_id_list") or [])
        kwargs = list(row.get("kwargs") or [])

        # Pool collection runs over the *raw* row regardless of whether
        # we keep it (more diversity).
        for iid, kw in zip(instruction_ids, kwargs):
            if not isinstance(kw, dict):
                continue
            if iid == "keywords:existence":
                for w in (kw.get("keywords") or []):
                    if isinstance(w, str) and w.strip():
                        existence_pool.add(w.strip())
            elif iid == "keywords:forbidden_words":
                for w in (kw.get("forbidden_words") or []):
                    if isinstance(w, str) and w.strip():
                        forbidden_pool.add(w.strip())
            elif iid == "keywords:frequency":
                w = kw.get("keyword")
                if isinstance(w, str) and w.strip():
                    frequency_pool.add(w.strip())

        if not instruction_ids:
            continue
        if not all(iid in SUPPORTED_INSTRUCTIONS for iid in instruction_ids):
            continue
        prompt = _safe_str(row.get("prompt"))
        if not prompt:
            continue
        cleaned_kwargs: List[Dict[str, Any]] = []
        for kw in kwargs:
            if not isinstance(kw, dict):
                cleaned_kwargs.append({})
                continue
            cleaned_kwargs.append({k: v for k, v in kw.items() if v is not None})
        out.append(
            {
                "prompt": prompt,
                "instruction_id_list": instruction_ids,
                "kwargs": cleaned_kwargs,
            }
        )
    # Stash pools as a side-channel attribute so main() can serialise
    # them after filtering — keeps the loader signature uniform.
    load_ifeval._pools = {
        "existence_keywords": sorted(existence_pool),
        "forbidden_words": sorted(forbidden_pool),
        "frequency_keywords": sorted(frequency_pool),
    }
    print(
        f"[preprocess] ifeval: {len(out)} items "
        f"(pools: existence={len(existence_pool)}, "
        f"forbidden={len(forbidden_pool)}, "
        f"frequency={len(frequency_pool)})",
        flush=True,
    )
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("gpqa", load_gpqa),
        ("mmlu_pro", load_mmlu_pro),
        ("hle", load_hle),
        ("ifeval", load_ifeval),
    ]

    manifest: List[Dict[str, Any]] = []
    cursor = 0
    for task_type, loader in sources:
        rows = loader()
        path = out_dir / f"{task_type}.jsonl"
        n = write_jsonl(path, rows)
        manifest.append(
            {
                "task_type": task_type,
                "start": cursor,
                "end": cursor + n,
                "count": n,
                "file": f"{task_type}.jsonl",
            }
        )
        cursor += n

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"total": cursor, "ranges": manifest}, f, indent=2)

    # IFEval keyword pools collected during loading.
    pools = getattr(load_ifeval, "_pools", None)
    if pools:
        with (out_dir / "ifeval_pools.json").open("w", encoding="utf-8") as f:
            json.dump(pools, f, indent=2, ensure_ascii=False)
        print(
            f"[preprocess] ifeval_pools.json: "
            f"existence={len(pools['existence_keywords'])} "
            f"forbidden={len(pools['forbidden_words'])} "
            f"frequency={len(pools['frequency_keywords'])}",
            flush=True,
        )

    print(f"[preprocess] total tasks: {cursor}", flush=True)
    for entry in manifest:
        print(
            f"[preprocess]   {entry['task_type']}: "
            f"[{entry['start']}, {entry['end']})  ({entry['count']} items)",
            flush=True,
        )


if __name__ == "__main__":
    main()

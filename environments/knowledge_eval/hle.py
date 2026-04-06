"""HLE multiple-choice task with anti-contamination modes.

HLE samples are pre-parsed into ``stem`` + ``options`` + ``answer`` at
build time (see ``preprocess.load_hle``).

Modes:
    ``off``              — original options, original order.
    ``shuffle``          — original options, deterministic shuffle.
    ``distractor_swap``  — falls back to ``shuffle``. HLE option counts
                           are variable (4-9) and the original
                           distractors are too question-specific to
                           usefully *replace*.
    ``cross_pool``       — original N options are kept and ``K`` extra
                           same-category options are appended (drawn
                           from a pool that contains both distractors
                           and other questions' correct answers, with
                           meta options like "None of the above"
                           filtered out at pool-build time). Yields an
                           N+K-way choice. The original difficulty is
                           preserved; memorising the answer text alone
                           no longer helps.
"""

import random
from typing import Any, Dict, Optional

from distractor_pool import DistractorPool
from labels import make_labels
from mc_extract import extract_choice
from models import Challenge

DEFAULT_EXTRA_DISTRACTORS = 4

PROMPT_TEMPLATE = """{question}

Answer Choices:
{options_block}

The last line of your response must be exactly "Answer: X" where X is one of {letters}."""


def _seeded_rng(task_id: int, perturb_seed: int, tag: str) -> random.Random:
    return random.Random(f"hle:{tag}:{task_id}:{perturb_seed}")


class HLETask:
    async def generate(
        self,
        sample: Dict[str, Any],
        task_id: int,
        *,
        mode: str = "shuffle",
        perturb_seed: int = 0,
        pool: Optional[DistractorPool] = None,
        extra_distractors: int = DEFAULT_EXTRA_DISTRACTORS,
    ) -> Challenge:
        stem = sample["stem"]
        options = list(sample["options"])
        gold_idx = ord(sample["answer"]) - ord("A")
        if not (0 <= gold_idx < len(options)):
            raise ValueError(
                f"HLE task {task_id}: gold letter {sample['answer']!r} "
                f"out of range for {len(options)} options"
            )
        correct = options[gold_idx]

        # distractor_swap doesn't make sense for HLE (option counts are
        # variable and originals are too question-specific to drop).
        # Treat it as plain shuffle.
        effective_mode = "shuffle" if mode == "distractor_swap" else mode

        if effective_mode == "cross_pool":
            if pool is None:
                raise ValueError("cross_pool mode requires a DistractorPool")
            choices = list(options)
            if extra_distractors > 0:
                extra_rng = _seeded_rng(task_id, perturb_seed, "cross_extra")
                extras = pool.sample_cross(
                    category=sample.get("category", "Unknown"),
                    exclude_idx=task_id,
                    forbidden_texts=choices,
                    n=extra_distractors,
                    rng=extra_rng,
                )
                choices.extend(extras)
        elif effective_mode == "off":
            choices = list(options)
        else:  # shuffle
            choices = list(options)

        if effective_mode != "off":
            shuffle_rng = _seeded_rng(task_id, perturb_seed, "shuffle")
            shuffle_rng.shuffle(choices)

        labels = make_labels(len(choices))
        correct_letter = labels[choices.index(correct)]
        options_block = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(choices))
        prompt = PROMPT_TEMPLATE.format(
            question=stem,
            options_block=options_block,
            letters="/".join(labels),
        )

        return Challenge(
            env="knowledge_eval:hle",
            prompt=prompt,
            extra={
                "correct_letter": correct_letter,
                "valid_letters": labels,
                "task_id": task_id,
                "mode": mode,
                "effective_mode": effective_mode,
                "perturb_seed": perturb_seed,
                "extra_distractors": extra_distractors if effective_mode == "cross_pool" else 0,
                "category": sample.get("category", ""),
                "id": sample.get("id", ""),
                "rendered_options": choices,
            },
        )

    async def evaluate(self, response: str, challenge: Challenge) -> float:
        valid = challenge.extra.get("valid_letters") or make_labels(5)
        guess = extract_choice(response or "", valid)
        if guess is None:
            return 0.0
        return 1.0 if guess == challenge.extra["correct_letter"] else 0.0

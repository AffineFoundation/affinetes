"""MMLU-Pro task with anti-contamination modes.

Modes mirror :mod:`gpqa`:
    ``off``              — original options, original order.
    ``shuffle``          — original options, deterministic shuffle.
    ``distractor_swap``  — incorrect options replaced with same-category
                           distractors from other questions, then shuffled.
                           Option count stays the same as the source row.
    ``cross_pool``       — original options are kept and ``K`` extra
                           cross-question options are appended, then
                           everything is shuffled. Yields an N+K-way
                           choice. Letters extend up to Z if needed.
"""

import random
from typing import Any, Dict, Optional

from distractor_pool import DistractorPool
from labels import make_labels
from mc_extract import extract_choice
from models import Challenge

DEFAULT_EXTRA_DISTRACTORS = 4

PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response must be exactly "Answer: X" where X is one of {letters}.

Question:
{question}

Options:
{options_block}
"""


def _seeded_rng(task_id: int, perturb_seed: int, tag: str) -> random.Random:
    return random.Random(f"mmlu_pro:{tag}:{task_id}:{perturb_seed}")


class MMLUProTask:
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
        options = list(sample["options"])
        gold_idx = ord(sample["answer"]) - ord("A")
        if not (0 <= gold_idx < len(options)):
            raise ValueError(
                f"MMLU-Pro task {task_id}: gold letter {sample['answer']!r} "
                f"out of range for {len(options)} options"
            )
        correct = options[gold_idx]
        distractors = [opt for i, opt in enumerate(options) if i != gold_idx]

        # ---- Build the choice list according to the mode ----
        if mode == "distractor_swap":
            if pool is None:
                raise ValueError("distractor_swap mode requires a DistractorPool")
            swap_rng = _seeded_rng(task_id, perturb_seed, "swap")
            swapped = pool.sample(
                category=sample.get("category", "Unknown"),
                exclude_idx=task_id,
                forbidden_texts=[correct, *distractors],
                n=len(distractors),
                rng=swap_rng,
            )
            while len(swapped) < len(distractors):
                for d in distractors:
                    if d not in swapped:
                        swapped.append(d)
                        if len(swapped) == len(distractors):
                            break
                else:
                    break
            choices = [correct, *swapped[: len(distractors)]]

        elif mode == "cross_pool":
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

        else:
            choices = list(options)

        # ---- Shuffle (everything except 'off') ----
        if mode != "off":
            shuffle_rng = _seeded_rng(task_id, perturb_seed, "shuffle")
            shuffle_rng.shuffle(choices)

        labels = make_labels(len(choices))
        correct_letter = labels[choices.index(correct)]
        options_block = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(choices))
        prompt = PROMPT_TEMPLATE.format(
            letters="/".join(labels),
            question=sample["question"],
            options_block=options_block,
        )

        return Challenge(
            env="knowledge_eval:mmlu_pro",
            prompt=prompt,
            extra={
                "correct_letter": correct_letter,
                "valid_letters": labels,
                "task_id": task_id,
                "mode": mode,
                "perturb_seed": perturb_seed,
                "extra_distractors": extra_distractors if mode == "cross_pool" else 0,
                "category": sample.get("category", ""),
                "rendered_options": choices,
            },
        )

    async def evaluate(self, response: str, challenge: Challenge) -> float:
        valid = challenge.extra.get("valid_letters") or make_labels(10)
        guess = extract_choice(response or "", valid)
        if guess is None:
            return 0.0
        return 1.0 if guess == challenge.extra["correct_letter"] else 0.0

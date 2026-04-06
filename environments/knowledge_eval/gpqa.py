"""GPQA diamond task with anti-contamination modes.

Modes:
    ``off``              — original distractors, original order.
    ``shuffle``          — original distractors, deterministic shuffle.
    ``distractor_swap``  — distractors replaced with same-category
                           distractors from other questions, then
                           shuffled. Option count stays at 4.
    ``cross_pool``       — original 4 options are kept and ``K`` extra
                           cross-question options (drawn from a pool
                           that includes both distractors *and* other
                           questions' correct answers) are appended,
                           then everything is shuffled. Yields a
                           4+K-way choice. The original difficulty is
                           preserved (the model still has to defeat
                           the carefully crafted distractors), but
                           memorising "the answer text" no longer
                           helps because the new options also look
                           like plausibly-correct answers from the
                           same domain.
"""

import random
from typing import Any, Dict, Optional

from distractor_pool import DistractorPool
from mc_extract import extract_choice
from models import Challenge

ALL_LETTERS = [chr(ord("A") + i) for i in range(26)]  # A..Z
N_DISTRACTORS = 3
DEFAULT_EXTRA_DISTRACTORS = 4

PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response must be exactly "Answer: X" where X is one of {letters}.

Question:
{question}

Options:
{options_block}
"""


def _seeded_rng(task_id: int, perturb_seed: int, tag: str) -> random.Random:
    return random.Random(f"gpqa:{tag}:{task_id}:{perturb_seed}")


class GPQATask:
    """GPQA evaluator with shuffle / distractor-swap / cross_pool support."""

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
        correct = sample["correct"]
        distractors = list(sample["distractors"])

        # ---- Build the choice list according to the mode ----
        if mode == "distractor_swap":
            if pool is None:
                raise ValueError("distractor_swap mode requires a DistractorPool")
            swap_rng = _seeded_rng(task_id, perturb_seed, "swap")
            swapped = pool.sample(
                category=sample.get("category", "Unknown"),
                exclude_idx=task_id,
                forbidden_texts=[correct, *distractors],
                n=N_DISTRACTORS,
                rng=swap_rng,
            )
            while len(swapped) < N_DISTRACTORS:
                for d in distractors:
                    if d not in swapped:
                        swapped.append(d)
                        if len(swapped) == N_DISTRACTORS:
                            break
                else:
                    break
            choices = [correct, *swapped[:N_DISTRACTORS]]

        elif mode == "cross_pool":
            if pool is None:
                raise ValueError("cross_pool mode requires a DistractorPool")
            choices = [correct, *distractors]
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
            choices = [correct, *distractors]

        # ---- Shuffle (everything except 'off') ----
        if mode != "off":
            shuffle_rng = _seeded_rng(task_id, perturb_seed, "shuffle")
            shuffle_rng.shuffle(choices)

        if len(choices) > len(ALL_LETTERS):
            raise ValueError(
                f"GPQA task {task_id}: too many options ({len(choices)}); "
                f"max is {len(ALL_LETTERS)}"
            )
        letters = ALL_LETTERS[: len(choices)]
        correct_letter = letters[choices.index(correct)]
        options_block = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        prompt = PROMPT_TEMPLATE.format(
            letters="/".join(letters),
            question=sample["question"],
            options_block=options_block,
        )

        return Challenge(
            env="knowledge_eval:gpqa",
            prompt=prompt,
            extra={
                "correct_letter": correct_letter,
                "valid_letters": letters,
                "task_id": task_id,
                "mode": mode,
                "perturb_seed": perturb_seed,
                "extra_distractors": extra_distractors if mode == "cross_pool" else 0,
                "rendered_options": choices,
            },
        )

    async def evaluate(self, response: str, challenge: Challenge) -> float:
        valid = set(challenge.extra.get("valid_letters", ALL_LETTERS[:4]))
        guess = extract_choice(response or "", valid)
        if guess is None:
            return 0.0
        return 1.0 if guess == challenge.extra["correct_letter"] else 0.0

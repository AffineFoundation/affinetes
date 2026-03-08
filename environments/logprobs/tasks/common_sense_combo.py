"""Task type 0: common_sense_combo

Generates an open-ended Q&A prompt by sampling N questions from the fact
pool and formatting them with a randomly chosen template. Model can answer
freely without any constrained format.

Sample space:
  ~49 total facts, choose 3-7 → easily covers 100M unique seeds
  via language × template × question selection × ordering.
"""

import random
import sys

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from tasks.base import BaseTask
from data.facts import ALL_FACTS, TEMPLATES_EN, TEMPLATES_ZH


class CommonSenseComboTask(BaseTask):
    """Combine N common-sense questions into an open-ended prompt."""

    def generate(self, seed: int):
        rng = random.Random(seed)

        language = rng.choice(["en", "zh"])
        n_questions = rng.randint(3, 7)

        # Sample without replacement, deterministically
        indices = list(range(len(ALL_FACTS)))
        rng.shuffle(indices)
        selected = [ALL_FACTS[i] for i in indices[:n_questions]]

        # Build numbered question list
        items_lines = []
        answers = []
        for i, (q_en, q_zh, a_en, a_zh) in enumerate(selected, 1):
            question = q_en if language == "en" else q_zh
            answer = a_en if language == "en" else a_zh
            items_lines.append(f"{i}. {question}")
            answers.append(answer)

        items_str = "\n".join(items_lines)

        templates = TEMPLATES_EN if language == "en" else TEMPLATES_ZH
        template = rng.choice(templates)
        prompt = template.format(items=items_str)

        metadata = {
            "seed": seed,
            "language": language,
            "n_questions": n_questions,
            "answers": answers,
        }
        return prompt, metadata

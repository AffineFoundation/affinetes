"""Task type 2: long_context

Generates long prompts by combining multiple text passages on diverse topics,
followed by an analysis question. Prompt length ranges from ~2000 to ~8000 tokens
depending on seed.

Sample space:
  14 passages, choose 8-14, 2 languages, 5 question templates
  → easily covers 100M unique seeds via selection × ordering × language × question.
"""

import random
import sys

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from tasks.base import BaseTask
from data.passages import PASSAGES, QUESTIONS_EN, QUESTIONS_ZH


class LongContextTask(BaseTask):
    """Combine multiple long passages into a reading comprehension prompt."""

    def generate(self, seed: int):
        rng = random.Random(seed)

        language = rng.choice(["en", "zh"])
        n_passages = rng.randint(8, len(PASSAGES))

        # Sample and shuffle passages
        indices = list(range(len(PASSAGES)))
        rng.shuffle(indices)
        selected = [PASSAGES[i] for i in indices[:n_passages]]

        # Build the long document
        sections = []
        topics = []
        for i, (topic, text_en, text_zh) in enumerate(selected, 1):
            text = text_en if language == "en" else text_zh
            sections.append(f"## Section {i}: {topic.replace('_', ' ').title()}\n\n{text}")
            topics.append(topic)

        document = "\n\n".join(sections)

        # Pick a question
        questions = QUESTIONS_EN if language == "en" else QUESTIONS_ZH
        question = rng.choice(questions)

        if language == "en":
            prompt = f"Read the following document carefully and then answer the question at the end.\n\n{document}\n\n---\n\nQuestion: {question}"
        else:
            prompt = f"请仔细阅读以下文档，然后回答最后的问题。\n\n{document}\n\n---\n\n问题：{question}"

        metadata = {
            "seed": seed,
            "language": language,
            "n_passages": n_passages,
            "topics": topics,
            "question": question,
        }
        return prompt, metadata

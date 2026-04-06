"""Open-ended base prompts for perturbed IFEval tasks.

Each prompt asks for a *generative*, free-form response on a topic
that is intentionally compatible with arbitrary instruction-following
constraints. Avoid prompts that prescribe a structure that would
clash with constraints (e.g. don't ask the model to "list five items
separated by commas" — that would conflict with ``no_comma`` or
``number_bullet_lists``).

The pool size sets the per-task variant count contributed by base
swap. Combined with kwargs perturbation it gives several thousand
distinct (base, kwargs) tuples per IFEval task.
"""

BASE_PROMPTS = [
    "Write a short essay about a topic that interests you.",
    "Describe your ideal weekend in detail.",
    "Tell a brief story about an adventurous journey.",
    "Explain a concept from physics to a curious teenager.",
    "Describe the process of brewing a perfect cup of tea.",
    "Write a paragraph about the importance of curiosity.",
    "Reflect on what makes a good friendship.",
    "Discuss the role of art in modern society.",
    "Describe a memorable meal you have had or imagined.",
    "Write about a hobby you find rewarding.",
    "Explain how the seasons change throughout the year.",
    "Describe an interesting scientific discovery in simple terms.",
    "Write a short biography of an imaginary inventor.",
    "Describe what a perfect garden would look like.",
    "Explain why kindness matters in everyday life.",
    "Write a short reflection on the value of lifelong learning.",
    "Describe an imaginary city in the year 2150.",
    "Write about a place you have always wanted to visit.",
    "Discuss the benefits of regular exercise.",
    "Describe a piece of music that makes you feel calm.",
    "Tell me about a fictional character you find admirable.",
    "Explain how a simple recipe is prepared from start to finish.",
    "Write a paragraph praising the role of teachers in society.",
    "Describe an unusual weather phenomenon you have heard about.",
    "Write about why reading books can be enjoyable.",
    "Discuss how technology has changed the way we communicate.",
    "Describe a dream you remember vividly.",
    "Write about an animal you find fascinating and why.",
    "Explain why honesty matters in close relationships.",
    "Describe what learning a new language feels like.",
    "Write a short message of encouragement to a stranger.",
    "Describe the most beautiful natural scene you can imagine.",
]

"""
Game of 24 Configuration
Simplified seed-based generation with flexible parameters
"""

import random

# ============================================================================
# Prompt Templates
# ============================================================================

# Chinese prompts (25 variants)
PROMPTS_ZH = [
    "用数字 {numbers} 和运算 {operators} 算出 {result}。",
    "给定数字 {numbers}，用 {operators} 得到 {result}。",
    "计算：{numbers} → {result}，运算符：{operators}。",
    "数字 {numbers}，目标 {result}，可用运算：{operators}。",
    "使用 {numbers} 通过 {operators} 计算 {result}。",
    "用 {numbers} 这几个数，通过 {operators} 运算得到 {result}。",
    "给定数字 {numbers}，使用算术运算 {operators} 得到 {result}。",
    "数字：{numbers}，运算符：{operators}，目标：{result}。",
    "请用 {numbers} 和 {operators} 计算出 {result}。",
    "用这些数字 {numbers} 通过 {operators} 算出 {result}。",
    "给你数字 {numbers}，用 {operators} 凑出 {result}。",
    "数字 {numbers}，想办法用 {operators} 得到 {result}。",
    "使用数字 {numbers}，运算符 {operators}，计算结果 {result}。",
    "用 {numbers}，通过 {operators} 运算，得到 {result}。",
    "给定 {numbers}，使用 {operators}，目标是 {result}。",
    "数字组合 {numbers}，运算 {operators}，目标值 {result}。",
    "利用 {numbers} 和 {operators} 计算 {result}。",
    "用这组数字 {numbers}，运算符 {operators}，算出 {result}。",
    "给定数字 {numbers}，通过 {operators} 得出 {result}。",
    "数字：{numbers}，可用运算：{operators}，目标：{result}。",
    "使用 {numbers}，运算符 {operators}，计算得到 {result}。",
    "用数字 {numbers}，通过算术运算 {operators}，得到 {result}。",
    "给你 {numbers}，用 {operators} 算出 {result}。",
    "数字 {numbers}，想办法通过 {operators} 计算出 {result}。",
    "用 {numbers} 这几个数字，运算 {operators}，目标 {result}。",
]

# English prompts (25 variants)
PROMPTS_EN = [
    "Use {numbers} with {operators} to get {result}.",
    "Given {numbers}, use {operators} to obtain {result}.",
    "Calculate: {numbers} → {result}, operators: {operators}.",
    "Numbers {numbers}, target {result}, available: {operators}.",
    "Use {numbers} and {operators} to compute {result}.",
    "Given the numbers {numbers}, apply {operators} to get {result}.",
    "Numbers: {numbers}, operators: {operators}, target: {result}.",
    "Use {numbers} with arithmetic operations {operators} to get {result}.",
    "Given {numbers}, use the operations {operators} to obtain {result}.",
    "Calculate {result} using {numbers} and {operators}.",
    "Using {numbers}, apply {operators} to reach {result}.",
    "Numbers {numbers}, find {result} using {operators}.",
    "Given the numbers {numbers}, apply the arithmetic operations {operators} to get the result of {result}.",
    "Use {numbers} and the operators {operators} to calculate {result}.",
    "With numbers {numbers} and operations {operators}, get {result}.",
    "Numbers: {numbers}, operations: {operators}, goal: {result}.",
    "Apply {operators} to {numbers} to obtain {result}.",
    "Using {numbers}, calculate {result} with {operators}.",
    "Given {numbers}, compute {result} using {operators}.",
    "Numbers {numbers}, target value {result}, operators {operators}.",
    "Use the numbers {numbers} with {operators} to reach {result}.",
    "Given {numbers}, use arithmetic {operators} to get {result}.",
    "Calculate {result} from {numbers} using {operators}.",
    "Numbers {numbers}, find a way to get {result} using {operators}.",
    "Using {numbers}, apply the operations {operators} to calculate {result}.",
]

# Answer instructions
ANSWER_INSTRUCTIONS = {
    "zh": "在回答的最后，请输出一个 ```python 代码块。代码块中仅包含一个代表答案的表达式，并且该表达式可以直接被 Python 中的 eval() 函数求值。如果无解，请输出 ```python\\nNone\\n```。",
    "en": "At the end of your response, please output a ```python code block. The code block should contain only a single expression representing the answer, which can be directly evaluated using Python's eval() function. If there is no solution, please output ```python\\nNone\\n```.",
}

# ============================================================================
# Parameter Derivation from Seed
# ============================================================================

def derive_params_from_seed(seed: int):
    """
    Derive all generation parameters from seed

    Returns:
        dict with keys: prompt_idx, min_val, max_val, num_count, target, operators
    """
    rng = random.Random(seed)

    # Prompt variant (0-49, alternates between zh and en)
    prompt_idx = seed % 50

    # Number range
    min_val = rng.choice([1, 2, 3, 5])
    max_val = rng.choice([9, 10, 12, 13, 15, 18, 20])

    # Number count (4, 5, or 6 numbers)
    num_count = rng.choice([4, 4, 4, 5, 5, 6])  # Weighted towards 4

    # Target value
    target = rng.choice([24, 36, 48, 60, 72, 100])

    # Operator set
    ops_sets = [
        ['+', '-', '*', '/'],  # Full set (most common)
        ['+', '-', '*', '/'],
        ['+', '-', '*', '/'],
        ['+', '-', '*'],       # No division
        ['+', '-'],            # Addition and subtraction only
    ]
    operators = rng.choice(ops_sets)

    return {
        'prompt_idx': prompt_idx,
        'min_val': min_val,
        'max_val': max_val,
        'num_count': num_count,
        'target': target,
        'operators': operators,
    }

def get_prompt_template(prompt_idx: int):
    """Get prompt template and language based on index"""
    if prompt_idx < 25:
        # Chinese prompts
        return PROMPTS_ZH[prompt_idx], 'zh'
    else:
        # English prompts
        return PROMPTS_EN[prompt_idx - 25], 'en'

def format_question(numbers, operators, target, prompt_idx):
    """Format the question using the appropriate template"""
    template, lang = get_prompt_template(prompt_idx)

    # Format numbers and operators for display
    numbers_str = str(numbers)
    operators_str = str(operators)

    # Create question
    question = template.format(
        numbers=numbers_str,
        operators=operators_str,
        result=target
    )

    # Add answer instruction
    question += "\n\n" + ANSWER_INSTRUCTIONS[lang]

    return question, lang

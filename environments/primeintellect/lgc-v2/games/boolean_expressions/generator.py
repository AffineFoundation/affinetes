"""
Boolean Expressions Task - Generator

Generates boolean expression evaluation problems with seed-based determinism.
"""

import random
import re
from typing import List, Tuple, Optional

from base.data import Data
from .config import (
    derive_params_from_seed,
    get_prompts,
    get_common_sense_facts,
    get_capitals,
    get_elements,
    get_days_in_month,
    get_planet_order,
    get_animal_legs,
    get_animal_classes,
    get_historical_events,
    get_states_of_matter,
    get_country_continent,
    get_country_currency,
    get_olympics_data,
    PRIMES_UNDER_100,
    PERFECT_SQUARES,
    FIBONACCI_NUMBERS,
    LEAP_YEARS,
    OPTION_LABELS,
)


class BooleanExpressionsGenerator:
    """Generate boolean expression evaluation challenges with seed-based determinism."""

    # All available fact types
    FACT_TYPES = [
        "common",       # Common sense facts
        "geography",    # Capitals
        "math",         # Prime, divisible, comparison, square, perfect_square, fibonacci, even_odd
        "time",         # Days in month, leap year
        "science",      # Element symbols
        "astronomy",    # Planet order
        "biology",      # Animal legs, classification
        "history",      # Historical events
        "physics",      # States of matter
        "geo_extended", # Continent
        "currency",     # Country currency
        "sports",       # Olympics
    ]

    def __init__(self):
        pass

    def generate(self, seed: int = None, **kwargs) -> Data:
        """
        Generate a single boolean expressions challenge based on seed.

        Args:
            seed: Random seed for deterministic generation
            **kwargs: Ignored, for compatibility

        Returns:
            Data object containing question, answer, and metadata
        """
        if seed is None:
            seed = random.randint(0, 99_999_999)

        # Derive all parameters from seed
        params = derive_params_from_seed(seed)

        # Create RNG for content generation
        rng = random.Random(seed)

        # Initialize language-specific data
        language = params["language"]
        self._init_language_data(language)

        # Generate expressions
        expressions = self._generate_expressions(
            rng,
            num_expressions=params["num_options"],
            depth=params["depth"],
        )

        # Build options and find true ones
        options = []
        true_options = []

        for i, (expr, value) in enumerate(expressions):
            if i >= len(OPTION_LABELS):
                break
            label = OPTION_LABELS[i]
            options.append(f"{label}. {expr}")
            if value:
                true_options.append(label)

        # Format question
        prompts = get_prompts(language)
        prompt_template = prompts[params["prompt_idx"]]
        question = prompt_template.format(options="\n".join(options))

        # Build answer
        true_options.sort()
        answer = ",".join(true_options)

        # Metadata
        metadata = {
            "seed": seed,
            "language": language,
            "depth": params["depth"],
            "num_options": params["num_options"],
            "true_count": len(true_options),
            "expressions": [(expr, val) for expr, val in expressions],
        }

        return Data(
            question=question,
            answer=answer,
            metadata=metadata,
        )

    def _init_language_data(self, language: str):
        """Initialize language-specific data."""
        self.language = language
        self.true_facts, self.false_facts = get_common_sense_facts(language)
        self.capitals = get_capitals(language)
        self.elements = get_elements(language)
        self.days_in_month = get_days_in_month(language)
        self.planet_order = get_planet_order(language)
        self.animal_legs = get_animal_legs(language)
        self.animal_classes = get_animal_classes(language)
        self.historical_events = get_historical_events(language)
        self.states_of_matter, self.state_names = get_states_of_matter(language)
        self.country_continent, self.continents = get_country_continent(language)
        self.country_currency = get_country_currency(language)
        self.olympics_data = get_olympics_data(language)

        # Comparison operators text mapping
        if language == "cn":
            self.text_to_op = {
                "大于": ">",
                "小于": "<",
                "大于等于": ">=",
                "小于等于": "<=",
                "等于": "==",
                "不等于": "!=",
            }
        else:
            self.text_to_op = {
                "is greater than": ">",
                "is less than": "<",
                "is greater than or equal to": ">=",
                "is less than or equal to": "<=",
                "is equal to": "==",
                "is not equal to": "!=",
            }

    def _generate_expressions(
        self,
        rng: random.Random,
        num_expressions: int,
        depth: int,
    ) -> List[Tuple[str, bool]]:
        """Generate multiple boolean expressions with their truth values."""
        expressions = []
        max_attempts = 100

        for _ in range(num_expressions):
            attempt = 0
            while attempt < max_attempts:
                expr = self._generate_boolean_expr(rng, depth)
                value = self._evaluate_expression(expr)

                # Ensure valid expression and reasonable length
                if value is not None and len(expr) <= 500:
                    expressions.append((expr, value))
                    break
                attempt += 1

            # Fallback if too many failures - generate simple fact
            if attempt >= max_attempts:
                expr = self._generate_leaf_expr(rng)
                value = self._evaluate_expression(expr)
                if value is None:
                    value = False
                expressions.append((expr, value))

        # Ensure at least one true option exists
        if not any(val for _, val in expressions):
            # Replace one with a simple true fact
            idx = rng.randint(0, len(expressions) - 1)
            true_fact = rng.choice(self.true_facts)
            expressions[idx] = (true_fact, True)

        return expressions

    def _generate_boolean_expr(self, rng: random.Random, depth: int) -> str:
        """Recursively generate a boolean expression."""
        # Base case: generate leaf node
        if depth <= 0 or (depth <= 2 and rng.random() < 0.3):
            return self._generate_leaf_expr(rng)

        choice = rng.random()

        if choice < 0.30:
            # NOT expression
            sub_expr = self._generate_boolean_expr(rng, depth - 1)
            not_count = rng.randint(1, 2)
            prefix = "not " * not_count
            return f"{prefix}({sub_expr})"

        elif choice < 0.65:
            # AND expression
            left = self._generate_boolean_expr(rng, depth - 1)
            right = self._generate_boolean_expr(rng, depth - 1)

            if rng.random() < 0.3 and depth > 2:
                middle = self._generate_boolean_expr(rng, depth - 2)
                return f"({left}) and ({middle}) and ({right})"
            return f"({left}) and ({right})"

        else:
            # OR expression
            left = self._generate_boolean_expr(rng, depth - 1)
            right = self._generate_boolean_expr(rng, depth - 1)

            if rng.random() < 0.3 and depth > 2:
                middle = self._generate_boolean_expr(rng, depth - 2)
                return f"({left}) or ({middle}) or ({right})"
            return f"({left}) or ({right})"

    def _generate_leaf_expr(self, rng: random.Random) -> str:
        """Generate a leaf expression (fact or comparison, no bare literals)."""
        choice = rng.random()

        if choice < 0.45:
            # Arithmetic comparison
            return self._generate_arithmetic_comparison(rng)
        else:
            # Fact expression
            return self._generate_fact_expr(rng)

    def _generate_arithmetic_comparison(self, rng: random.Random) -> str:
        """Generate an arithmetic comparison expression."""
        comparison_ops = [">", "<", ">=", "<=", "==", "!="]

        # Generate left and right arithmetic expressions
        left_expr = self._generate_arithmetic_expr(rng)
        right_expr = self._generate_arithmetic_expr(rng)

        # Choose comparison operator
        cmp_op = rng.choice(comparison_ops)

        # Occasionally use chain comparison (e.g., 1 < 2 < 3)
        if rng.random() < 0.15:
            middle_expr = self._generate_arithmetic_expr(rng)
            chain_op = rng.choice(["<", "<="])
            return f"({left_expr}) {chain_op} ({middle_expr}) {chain_op} ({right_expr})"

        return f"({left_expr}) {cmp_op} ({right_expr})"

    def _generate_arithmetic_expr(self, rng: random.Random) -> str:
        """Generate an arithmetic expression."""
        ops = ["+", "-", "*"]

        # Simple expression: a op b
        if rng.random() < 0.6:
            a = rng.randint(-20, 20)
            b = rng.randint(-20, 20)
            op = rng.choice(ops)
            return f"{a} {op} {b}"

        # More complex: a * b op c * d
        if rng.random() < 0.5:
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            c = rng.randint(-10, 10)
            d = rng.randint(-10, 10)
            op = rng.choice(ops)
            return f"{a} * {b} {op} {c} * {d}"

        # With power or modulo
        if rng.random() < 0.5:
            base = rng.randint(1, 5)
            exp = rng.randint(1, 3)
            return f"{base} ** {exp}"
        else:
            a = rng.randint(1, 50)
            b = rng.randint(1, 10)
            return f"{a} % {b}"

    def _generate_fact_expr(self, rng: random.Random) -> str:
        """Generate a fact expression from various categories."""
        fact_type = rng.choice(self.FACT_TYPES)

        if fact_type == "common":
            return self._generate_common_sense_fact(rng)
        elif fact_type == "geography":
            return self._generate_geography_fact(rng)
        elif fact_type == "math":
            return self._generate_math_fact(rng)
        elif fact_type == "time":
            return self._generate_time_fact(rng)
        elif fact_type == "science":
            return self._generate_science_fact(rng)
        elif fact_type == "astronomy":
            return self._generate_astronomy_fact(rng)
        elif fact_type == "biology":
            return self._generate_biology_fact(rng)
        elif fact_type == "history":
            return self._generate_history_fact(rng)
        elif fact_type == "physics":
            return self._generate_physics_fact(rng)
        elif fact_type == "geo_extended":
            return self._generate_geo_extended_fact(rng)
        elif fact_type == "currency":
            return self._generate_currency_fact(rng)
        elif fact_type == "sports":
            return self._generate_sports_fact(rng)
        else:
            return self._generate_common_sense_fact(rng)

    def _generate_common_sense_fact(self, rng: random.Random) -> str:
        """Generate a common sense fact."""
        if rng.random() < 0.5:
            return rng.choice(self.true_facts)
        else:
            return rng.choice(self.false_facts)

    def _generate_geography_fact(self, rng: random.Random) -> str:
        """Generate a geography fact about capitals."""
        countries = list(self.capitals.keys())
        country = rng.choice(countries)
        correct_capital = self.capitals[country]

        if rng.random() < 0.5:
            capital = correct_capital
        else:
            other_capitals = [c for c in self.capitals.values() if c != correct_capital]
            capital = rng.choice(other_capitals)

        if self.language == "cn":
            return f"{country}的首都是{capital}。"
        else:
            return f"The capital of {country} is {capital}."

    def _generate_math_fact(self, rng: random.Random) -> str:
        """Generate a mathematical fact."""
        math_type = rng.choice(["prime", "divisible", "comparison", "square", "perfect_square", "fibonacci", "even_odd"])

        if math_type == "prime":
            n = rng.randint(2, 99)
            if self.language == "cn":
                return f"{n}是质数。"
            else:
                return f"{n} is a prime number."

        elif math_type == "divisible":
            a = rng.randint(10, 100)
            b = rng.randint(2, 12)
            if self.language == "cn":
                return f"{a}能被{b}整除。"
            else:
                return f"{a} is divisible by {b}."

        elif math_type == "comparison":
            a = rng.randint(1, 100)
            b = rng.randint(1, 100)
            if rng.random() < 0.5:
                if self.language == "cn":
                    return f"{a}大于{b}。"
                else:
                    return f"{a} is greater than {b}."
            else:
                if self.language == "cn":
                    return f"{a}小于等于{b}。"
                else:
                    return f"{a} is less than or equal to {b}."

        elif math_type == "square":
            n = rng.randint(1, 15)
            square = n * n
            if rng.random() < 0.5:
                claimed_root = n
            else:
                claimed_root = rng.randint(1, 15)

            if self.language == "cn":
                return f"{square}的平方根是{claimed_root}。"
            else:
                return f"The square root of {square} is {claimed_root}."

        elif math_type == "perfect_square":
            n = rng.randint(1, 400)
            if self.language == "cn":
                return f"{n}是完全平方数。"
            else:
                return f"{n} is a perfect square."

        elif math_type == "fibonacci":
            n = rng.randint(1, 1000)
            if self.language == "cn":
                return f"{n}是斐波那契数。"
            else:
                return f"{n} is a Fibonacci number."

        else:  # even_odd
            n = rng.randint(1, 200)
            if rng.random() < 0.5:
                if self.language == "cn":
                    return f"{n}是偶数。"
                else:
                    return f"{n} is an even number."
            else:
                if self.language == "cn":
                    return f"{n}是奇数。"
                else:
                    return f"{n} is an odd number."

    def _generate_time_fact(self, rng: random.Random) -> str:
        """Generate a time-related fact."""
        time_type = rng.choice(["days_in_month", "leap_year"])

        if time_type == "days_in_month":
            month = rng.choice(list(self.days_in_month.keys()))
            correct_days = self.days_in_month[month]

            if rng.random() < 0.5:
                days = correct_days
            else:
                days = rng.choice([28, 29, 30, 31])

            if self.language == "cn":
                return f"{month}有{days}天。"
            else:
                return f"There are {days} days in {month}."

        else:  # leap_year
            year = rng.randint(1900, 2100)

            if self.language == "cn":
                return f"{year}年是闰年。"
            else:
                return f"{year} is a leap year."

    def _generate_science_fact(self, rng: random.Random) -> str:
        """Generate a science fact about elements."""
        elements_list = list(self.elements.keys())
        element = rng.choice(elements_list)
        correct_symbol = self.elements[element]

        if rng.random() < 0.5:
            symbol = correct_symbol
        else:
            other_symbols = [s for s in self.elements.values() if s != correct_symbol]
            symbol = rng.choice(other_symbols)

        if self.language == "cn":
            return f"{element}的化学符号是{symbol}。"
        else:
            return f"The chemical symbol for {element} is {symbol}."

    def _generate_astronomy_fact(self, rng: random.Random) -> str:
        """Generate an astronomy fact about planet order."""
        planet = rng.choice(list(self.planet_order.keys()))
        correct_order = self.planet_order[planet]

        if rng.random() < 0.5:
            order = correct_order
        else:
            order = rng.randint(1, 8)

        if self.language == "cn":
            return f"{planet}是距离太阳第{order}近的行星。"
        else:
            return f"{planet} is the {self._ordinal(order)} planet from the Sun."

    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal string."""
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _generate_biology_fact(self, rng: random.Random) -> str:
        """Generate a biology fact."""
        bio_type = rng.choice(["legs", "classification"])

        if bio_type == "legs":
            animal = rng.choice(list(self.animal_legs.keys()))
            correct_legs = self.animal_legs[animal]

            if rng.random() < 0.5:
                legs = correct_legs
            else:
                legs = rng.choice([0, 2, 4, 6, 8, 10, 100])

            if self.language == "cn":
                return f"{animal}有{legs}条腿。"
            else:
                return f"A {animal} has {legs} legs."

        else:  # classification
            class_type = rng.choice(["mammals", "birds", "reptiles", "fish", "insects"])
            class_set = self.animal_classes[class_type]
            animal = rng.choice(list(class_set))

            # Class name in target language
            class_names = {
                "mammals": ("哺乳动物", "mammal"),
                "birds": ("鸟类", "bird"),
                "reptiles": ("爬行动物", "reptile"),
                "fish": ("鱼类", "fish"),
                "insects": ("昆虫", "insect"),
            }

            if rng.random() < 0.5:
                claimed_class = class_type
            else:
                claimed_class = rng.choice(["mammals", "birds", "reptiles", "fish", "insects"])

            cn_name, en_name = class_names[claimed_class]

            if self.language == "cn":
                return f"{animal}是{cn_name}。"
            else:
                article = "an" if en_name[0] in "aeiou" else "a"
                return f"A {animal} is {article} {en_name}."

    def _generate_history_fact(self, rng: random.Random) -> str:
        """Generate a historical fact."""
        event = rng.choice(list(self.historical_events.keys()))
        correct_year = self.historical_events[event]

        if rng.random() < 0.5:
            year = correct_year
        else:
            # Generate a wrong year within reasonable range
            year = correct_year + rng.choice([-10, -5, -2, -1, 1, 2, 5, 10])

        if self.language == "cn":
            return f"{event}发生在{year}年。"
        else:
            return f"{event} occurred in {year}."

    def _generate_physics_fact(self, rng: random.Random) -> str:
        """Generate a physics fact about states of matter."""
        substance = rng.choice(list(self.states_of_matter.keys()))
        correct_state = self.states_of_matter[substance]

        if rng.random() < 0.5:
            state = correct_state
        else:
            state = rng.choice(self.state_names)

        if self.language == "cn":
            return f"{substance}在常温下是{state}。"
        else:
            return f"{substance.capitalize()} is a {state} at room temperature."

    def _generate_geo_extended_fact(self, rng: random.Random) -> str:
        """Generate extended geography facts about continents."""
        country = rng.choice(list(self.country_continent.keys()))
        correct_continent = self.country_continent[country]

        if rng.random() < 0.5:
            continent = correct_continent
        else:
            continent = rng.choice(self.continents)

        if self.language == "cn":
            return f"{country}位于{continent}。"
        else:
            return f"{country} is in {continent}."

    def _generate_currency_fact(self, rng: random.Random) -> str:
        """Generate a currency fact."""
        country = rng.choice(list(self.country_currency.keys()))
        correct_currency = self.country_currency[country]

        if rng.random() < 0.5:
            currency = correct_currency
        else:
            other_currencies = [c for c in self.country_currency.values() if c != correct_currency]
            currency = rng.choice(other_currencies)

        if self.language == "cn":
            return f"{country}的货币是{currency}。"
        else:
            return f"The currency of {country} is the {currency}."

    def _generate_sports_fact(self, rng: random.Random) -> str:
        """Generate a sports/Olympics fact."""
        year = rng.choice(list(self.olympics_data.keys()))
        correct_city = self.olympics_data[year]

        if rng.random() < 0.5:
            city = correct_city
        else:
            other_cities = [c for c in self.olympics_data.values() if c != correct_city]
            city = rng.choice(other_cities)

        if self.language == "cn":
            return f"{year}年奥运会在{city}举办。"
        else:
            return f"The {year} Olympics were held in {city}."

    def _evaluate_expression(self, expr: str) -> Optional[bool]:
        """Evaluate a boolean expression, returning None if evaluation fails."""
        processed = self._preprocess_expression(expr)
        try:
            result = eval(processed)
            return bool(result)
        except Exception:
            return None

    def _preprocess_expression(self, expr: str) -> str:
        """Preprocess expression for evaluation."""
        # Replace fact statements with their truth values
        expr = self._replace_facts(expr)

        # Replace text operators with symbols
        sorted_text_ops = sorted(self.text_to_op.keys(), key=len, reverse=True)
        for text in sorted_text_ops:
            op = self.text_to_op[text]
            expr = expr.replace(text, op)

        return expr

    def _replace_facts(self, expr: str) -> str:
        """Replace fact statements with True/False."""
        # Common sense facts
        for fact in self.true_facts:
            expr = expr.replace(fact, "True")
        for fact in self.false_facts:
            expr = expr.replace(fact, "False")

        # Geography facts (capitals)
        expr = self._replace_geography_facts(expr)

        # Math facts
        expr = self._replace_math_facts(expr)

        # Time facts
        expr = self._replace_time_facts(expr)

        # Science facts (elements)
        expr = self._replace_science_facts(expr)

        # Astronomy facts
        expr = self._replace_astronomy_facts(expr)

        # Biology facts
        expr = self._replace_biology_facts(expr)

        # History facts
        expr = self._replace_history_facts(expr)

        # Physics facts
        expr = self._replace_physics_facts(expr)

        # Extended geography facts
        expr = self._replace_geo_extended_facts(expr)

        # Currency facts
        expr = self._replace_currency_facts(expr)

        # Sports facts
        expr = self._replace_sports_facts(expr)

        return expr

    def _replace_geography_facts(self, expr: str) -> str:
        """Replace geography facts with True/False."""
        for country, capital in self.capitals.items():
            if self.language == "cn":
                true_fact = f"{country}的首都是{capital}。"
            else:
                true_fact = f"The capital of {country} is {capital}."
            expr = expr.replace(true_fact, "True")

        # Replace remaining as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+的首都是[\u4e00-\u9fa5]+。"
        else:
            pattern = r"The capital of [A-Za-z\s]+ is [A-Za-z\s]+\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_math_facts(self, expr: str) -> str:
        """Replace math facts with True/False."""
        # Prime numbers
        if self.language == "cn":
            pattern = r"(\d+)是质数。"
        else:
            pattern = r"(\d+) is a prime number\."

        def check_prime(match):
            n = int(match.group(1))
            return "True" if n in PRIMES_UNDER_100 else "False"
        expr = re.sub(pattern, check_prime, expr)

        # Divisibility
        if self.language == "cn":
            pattern = r"(\d+)能被(\d+)整除。"
        else:
            pattern = r"(\d+) is divisible by (\d+)\."

        def check_divisible(match):
            a, b = int(match.group(1)), int(match.group(2))
            return "True" if b != 0 and a % b == 0 else "False"
        expr = re.sub(pattern, check_divisible, expr)

        # Greater than
        if self.language == "cn":
            pattern = r"(\d+)大于(\d+)。"
        else:
            pattern = r"(\d+) is greater than (\d+)\."

        def check_greater(match):
            a, b = int(match.group(1)), int(match.group(2))
            return "True" if a > b else "False"
        expr = re.sub(pattern, check_greater, expr)

        # Less than or equal
        if self.language == "cn":
            pattern = r"(\d+)小于等于(\d+)。"
        else:
            pattern = r"(\d+) is less than or equal to (\d+)\."

        def check_leq(match):
            a, b = int(match.group(1)), int(match.group(2))
            return "True" if a <= b else "False"
        expr = re.sub(pattern, check_leq, expr)

        # Square root
        if self.language == "cn":
            pattern = r"(\d+)的平方根是(\d+)。"
        else:
            pattern = r"The square root of (\d+) is (\d+)\."

        def check_sqrt(match):
            n, root = int(match.group(1)), int(match.group(2))
            return "True" if root * root == n else "False"
        expr = re.sub(pattern, check_sqrt, expr)

        # Perfect square
        if self.language == "cn":
            pattern = r"(\d+)是完全平方数。"
        else:
            pattern = r"(\d+) is a perfect square\."

        def check_perfect_square(match):
            n = int(match.group(1))
            return "True" if n in PERFECT_SQUARES else "False"
        expr = re.sub(pattern, check_perfect_square, expr)

        # Fibonacci
        if self.language == "cn":
            pattern = r"(\d+)是斐波那契数。"
        else:
            pattern = r"(\d+) is a Fibonacci number\."

        def check_fibonacci(match):
            n = int(match.group(1))
            return "True" if n in FIBONACCI_NUMBERS else "False"
        expr = re.sub(pattern, check_fibonacci, expr)

        # Even number
        if self.language == "cn":
            pattern = r"(\d+)是偶数。"
        else:
            pattern = r"(\d+) is an even number\."

        def check_even(match):
            n = int(match.group(1))
            return "True" if n % 2 == 0 else "False"
        expr = re.sub(pattern, check_even, expr)

        # Odd number
        if self.language == "cn":
            pattern = r"(\d+)是奇数。"
        else:
            pattern = r"(\d+) is an odd number\."

        def check_odd(match):
            n = int(match.group(1))
            return "True" if n % 2 == 1 else "False"
        expr = re.sub(pattern, check_odd, expr)

        return expr

    def _replace_time_facts(self, expr: str) -> str:
        """Replace time facts with True/False."""
        # Days in month
        if self.language == "cn":
            pattern = r"([\u4e00-\u9fa5]+月)有(\d+)天。"
        else:
            pattern = r"There are (\d+) days in ([A-Za-z]+)\."

        def check_days(match):
            if self.language == "cn":
                month, days = match.group(1), int(match.group(2))
            else:
                days, month = int(match.group(1)), match.group(2)
            correct = self.days_in_month.get(month, -1)
            return "True" if days == correct else "False"
        expr = re.sub(pattern, check_days, expr)

        # Leap year
        if self.language == "cn":
            pattern = r"(\d+)年是闰年。"
        else:
            pattern = r"(\d+) is a leap year\."

        def check_leap(match):
            year = int(match.group(1))
            return "True" if year in LEAP_YEARS else "False"
        expr = re.sub(pattern, check_leap, expr)

        return expr

    def _replace_science_facts(self, expr: str) -> str:
        """Replace science facts with True/False."""
        for element, symbol in self.elements.items():
            if self.language == "cn":
                true_fact = f"{element}的化学符号是{symbol}。"
            else:
                true_fact = f"The chemical symbol for {element} is {symbol}."
            expr = expr.replace(true_fact, "True")

        # Replace remaining as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+的化学符号是[A-Za-z]+。"
        else:
            pattern = r"The chemical symbol for [A-Za-z]+ is [A-Za-z]+\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_astronomy_facts(self, expr: str) -> str:
        """Replace astronomy facts with True/False."""
        # Planet order
        for planet, order in self.planet_order.items():
            if self.language == "cn":
                true_fact = f"{planet}是距离太阳第{order}近的行星。"
            else:
                true_fact = f"{planet} is the {self._ordinal(order)} planet from the Sun."
            expr = expr.replace(true_fact, "True")

        # Replace remaining planet order as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+是距离太阳第\d+近的行星。"
        else:
            pattern = r"[A-Za-z]+ is the \d+(?:st|nd|rd|th) planet from the Sun\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_biology_facts(self, expr: str) -> str:
        """Replace biology facts with True/False."""
        # Animal legs
        for animal, legs in self.animal_legs.items():
            if self.language == "cn":
                true_fact = f"{animal}有{legs}条腿。"
            else:
                true_fact = f"A {animal} has {legs} legs."
            expr = expr.replace(true_fact, "True")

        # Replace remaining legs as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+有\d+条腿。"
        else:
            pattern = r"A [a-z]+ has \d+ legs\."
        expr = re.sub(pattern, "False", expr)

        # Animal classification
        class_names = {
            "mammals": ("哺乳动物", "mammal"),
            "birds": ("鸟类", "bird"),
            "reptiles": ("爬行动物", "reptile"),
            "fish": ("鱼类", "fish"),
            "insects": ("昆虫", "insect"),
        }

        for class_type, animals in self.animal_classes.items():
            cn_name, en_name = class_names[class_type]
            for animal in animals:
                if self.language == "cn":
                    true_fact = f"{animal}是{cn_name}。"
                else:
                    article = "an" if en_name[0] in "aeiou" else "a"
                    true_fact = f"A {animal} is {article} {en_name}."
                expr = expr.replace(true_fact, "True")

        # Replace remaining classification as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+是(?:哺乳动物|鸟类|爬行动物|鱼类|昆虫)。"
        else:
            pattern = r"A [a-z]+ is (?:a|an) (?:mammal|bird|reptile|fish|insect)\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_history_facts(self, expr: str) -> str:
        """Replace history facts with True/False."""
        for event, year in self.historical_events.items():
            if self.language == "cn":
                true_fact = f"{event}发生在{year}年。"
            else:
                true_fact = f"{event} occurred in {year}."
            expr = expr.replace(true_fact, "True")

        # Replace remaining as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5A-Za-z]+发生在\d+年。"
        else:
            pattern = r".+ occurred in \d+\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_physics_facts(self, expr: str) -> str:
        """Replace physics facts with True/False."""
        for substance, state in self.states_of_matter.items():
            if self.language == "cn":
                true_fact = f"{substance}在常温下是{state}。"
            else:
                true_fact = f"{substance.capitalize()} is a {state} at room temperature."
            expr = expr.replace(true_fact, "True")

        # Replace remaining as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+在常温下是(?:固体|液体|气体)。"
        else:
            pattern = r"[A-Za-z\s]+ is a (?:solid|liquid|gas) at room temperature\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_geo_extended_facts(self, expr: str) -> str:
        """Replace extended geography facts with True/False."""
        # Continent
        for country, continent in self.country_continent.items():
            if self.language == "cn":
                true_fact = f"{country}位于{continent}。"
            else:
                true_fact = f"{country} is in {continent}."
            expr = expr.replace(true_fact, "True")

        # Replace remaining continent as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+位于[\u4e00-\u9fa5]+。"
        else:
            pattern = r"[A-Za-z\s]+ is in [A-Za-z\s]+\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_currency_facts(self, expr: str) -> str:
        """Replace currency facts with True/False."""
        for country, currency in self.country_currency.items():
            if self.language == "cn":
                true_fact = f"{country}的货币是{currency}。"
            else:
                true_fact = f"The currency of {country} is the {currency}."
            expr = expr.replace(true_fact, "True")

        # Replace remaining as False
        if self.language == "cn":
            pattern = r"[\u4e00-\u9fa5]+的货币是[\u4e00-\u9fa5]+。"
        else:
            pattern = r"The currency of [A-Za-z\s]+ is the [A-Za-z]+\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def _replace_sports_facts(self, expr: str) -> str:
        """Replace sports facts with True/False."""
        for year, city in self.olympics_data.items():
            if self.language == "cn":
                true_fact = f"{year}年奥运会在{city}举办。"
            else:
                true_fact = f"The {year} Olympics were held in {city}."
            expr = expr.replace(true_fact, "True")

        # Replace remaining as False
        if self.language == "cn":
            pattern = r"\d+年奥运会在[\u4e00-\u9fa5]+举办。"
        else:
            pattern = r"The \d+ Olympics were held in [A-Za-z\s]+\."
        expr = re.sub(pattern, "False", expr)

        return expr

    def extract_answer(self, test_solution: str) -> Optional[str]:
        """
        Extract answer from model response.

        Looks for \\boxed{} pattern and extracts content.
        """
        last_box_index = test_solution.rfind("\\boxed{")
        if last_box_index == -1:
            return None

        start_index = last_box_index + len("\\boxed{")
        bracket_stack = 1
        end_index = start_index

        while end_index < len(test_solution) and bracket_stack > 0:
            if test_solution[end_index] == "{":
                bracket_stack += 1
            elif test_solution[end_index] == "}":
                bracket_stack -= 1
            end_index += 1

        if bracket_stack != 0:
            return None

        content = test_solution[start_index:end_index - 1].strip()
        return content

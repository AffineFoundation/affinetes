"""
Operation Generator - Symbol Operations Task

Generates custom operator definition and evaluation problems
"""

import random
import re
import uuid

from base.data import Data


class OperationGenerator:
    """Generate symbol operation challenges with seed-based determinism"""

    def __init__(self):
        # Available custom symbols
        self.symbols = ["△", "▽", "◇", "○", "☆", "◎", "□", "♡", "♢", "⊕", "⊗", "⊙"]

        # Base arithmetic operations
        self.base_operations = ["+", "-", "*", "/", "%", "**"]

        # Condition types (Chinese)
        self.condition_types = [
            "x和y都是偶数",
            "x和y都是奇数",
            "x是偶数",
            "x是奇数",
            "y是偶数",
            "y是奇数",
            "x大于y",
            "x小于y",
            "x等于y",
            "x是3的倍数",
            "y是3的倍数",
            "x是5的倍数",
            "y是5的倍数",
            "x和y的和是偶数",
            "x和y的和是奇数",
        ]

        # Condition to English mapping
        self.condition2english = {
            "x和y都是偶数": "x and y are both even",
            "x和y都是奇数": "x and y are both odd",
            "x是偶数": "x is even",
            "x是奇数": "x is odd",
            "y是偶数": "y is even",
            "y是奇数": "y is odd",
            "x大于y": "x is greater than y",
            "x小于y": "x is less than y",
            "x等于y": "x is equal to y",
            "x是3的倍数": "x is a multiple of 3",
            "y是3的倍数": "y is a multiple of 3",
            "x是5的倍数": "x is a multiple of 5",
            "y是5的倍数": "y is a multiple of 5",
            "x和y的和是偶数": "the sum of x and y is even",
            "x和y的和是奇数": "the sum of x and y is odd",
        }

        # Condition check functions
        self.condition_checks = {
            "x和y都是偶数": lambda x, y: x % 2 == 0 and y % 2 == 0,
            "x和y都是奇数": lambda x, y: x % 2 != 0 and y % 2 != 0,
            "x是偶数": lambda x, y: x % 2 == 0,
            "x是奇数": lambda x, y: x % 2 != 0,
            "y是偶数": lambda x, y: y % 2 == 0,
            "y是奇数": lambda x, y: y % 2 != 0,
            "x大于y": lambda x, y: x > y,
            "x小于y": lambda x, y: x < y,
            "x等于y": lambda x, y: x == y,
            "x是3的倍数": lambda x, y: x % 3 == 0,
            "y是3的倍数": lambda x, y: y % 3 == 0,
            "x是5的倍数": lambda x, y: x % 5 == 0,
            "y是5的倍数": lambda x, y: y % 5 == 0,
            "x和y的和是偶数": lambda x, y: (x + y) % 2 == 0,
            "x和y的和是奇数": lambda x, y: (x + y) % 2 != 0,
        }

    def generate(self, seed: int = None, **kwargs):
        """
        Generate a single operation task based on seed

        All parameters are derived from seed for determinism

        Args:
            seed: Random seed for deterministic generation
            **kwargs: Config parameters (ignored)

        Returns:
            Data object containing question, answer, and metadata
        """
        if seed is None:
            seed = random.randint(0, 99999999)

        # Derive parameters from seed
        rng = random.Random(seed)

        # Derive task parameters
        num_symbols = self._derive_num_symbols(rng)
        condition_rate = self._derive_condition_rate(rng)
        bracket_rate = self._derive_bracket_rate(rng)
        abs_rate = self._derive_abs_rate(rng)
        max_operands = self._derive_max_operands(rng)
        definition_num_symbols_max = self._derive_definition_num_symbols_max(rng)
        language = self._derive_language(rng)

        # Try to generate a valid problem
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                # Use attempt-modified seed for retries
                attempt_rng = random.Random(seed + attempt)

                result = self._generate_problem(
                    attempt_rng,
                    num_symbols=num_symbols,
                    condition_rate=condition_rate,
                    bracket_rate=bracket_rate,
                    abs_rate=abs_rate,
                    max_operands=max_operands,
                    definition_num_symbols_max=definition_num_symbols_max,
                    language=language
                )

                if result is not None:
                    # Check result is reasonable
                    try:
                        answer_val = float(result["answer"])
                        if abs(answer_val) < 100000:
                            return Data(
                                question=result["question"],
                                answer=result["answer"],
                                metadata={
                                    "seed": seed,
                                    "trace_id": str(uuid.uuid4()),
                                    "expression": result["metadata"]["expression"],
                                    "symbol_definitions": result["metadata"]["symbol_definitions"],
                                    "simplified_expr": result["metadata"]["simplified_expr"],
                                    "language": language,
                                    "num_symbols": num_symbols,
                                }
                            )
                    except (ValueError, TypeError):
                        continue

            except Exception:
                continue

        # If all attempts fail, return a simple fallback
        return self._generate_fallback(seed, rng, language)

    def _derive_num_symbols(self, rng):
        """Derive number of symbols from seed (1-3)"""
        return rng.randint(1, 3)

    def _derive_condition_rate(self, rng):
        """Derive condition rate from seed"""
        return rng.choice([0.3, 0.4, 0.5, 0.6])

    def _derive_bracket_rate(self, rng):
        """Derive bracket rate from seed"""
        return rng.choice([0.2, 0.3, 0.4])

    def _derive_abs_rate(self, rng):
        """Derive abs rate from seed"""
        return rng.choice([0.3, 0.4, 0.5])

    def _derive_max_operands(self, rng):
        """Derive max operands from seed"""
        return rng.randint(4, 6)

    def _derive_definition_num_symbols_max(self, rng):
        """Derive max symbols in definition from seed"""
        return rng.randint(3, 5)

    def _derive_language(self, rng):
        """Derive language from seed"""
        return rng.choice(["chinese", "english"])

    def _generate_problem(self, rng, num_symbols, condition_rate, bracket_rate,
                         abs_rate, max_operands, definition_num_symbols_max, language):
        """Generate a single operation problem"""
        # Select symbols
        selected_symbols = rng.sample(self.symbols, num_symbols)

        # Create symbol definitions
        symbol_definitions = {}
        for symbol in selected_symbols:
            symbol_definitions[symbol] = self._create_symbol_definition(
                rng, symbol_definitions, condition_rate, bracket_rate,
                abs_rate, definition_num_symbols_max
            )

        # Assign precedence
        for symbol in selected_symbols:
            precedence = rng.randint(1, 5)
            symbol_definitions[symbol]["precedence"] = precedence

        # Generate expression
        expression, operands = self._generate_expression(rng, list(symbol_definitions.keys()), max_operands)

        # Evaluate expression
        result, simplified_expr = self._evaluate_expression(expression, symbol_definitions)

        if isinstance(result, str) and (result == "evaluation_timeout" or result == "evaluation_error"):
            return None

        # Format question
        question = self._format_question(expression, symbol_definitions, language)

        return {
            "question": question,
            "answer": str(result),
            "metadata": {
                "expression": expression,
                "symbol_definitions": symbol_definitions,
                "result": result,
                "simplified_expr": simplified_expr,
            }
        }

    def _create_symbol_definition(self, rng, symbol_definitions, condition_rate,
                                  bracket_rate, abs_rate, definition_num_symbols_max):
        """Create a symbol definition with optional conditions"""
        definition = {"conditions": [], "associativity": "left", "precedence": 0}

        if rng.random() < condition_rate:
            selected_condition = rng.choice(self.condition_types)
            operation = self._generate_random_operation(
                rng, symbol_definitions, bracket_rate, abs_rate, definition_num_symbols_max
            )
            definition["conditions"].append({
                "condition": selected_condition,
                "operation": operation
            })

        definition["default_operation"] = self._generate_random_operation(
            rng, symbol_definitions, bracket_rate, abs_rate, definition_num_symbols_max
        )

        return definition

    def _generate_random_operation(self, rng, symbol_definitions, bracket_rate,
                                   abs_rate, definition_num_symbols_max):
        """Generate a random operation expression"""
        basic_ops = ["+", "-", "*", "/", "**"]
        weights = [0.3, 0.3, 0.25, 0.1, 0.05]
        custom_symbols = list(symbol_definitions.keys())

        max_attempts = 50
        for _ in range(max_attempts):
            try:
                num_variables = rng.randint(2, definition_num_symbols_max)
                num_constants = rng.randint(0, definition_num_symbols_max - num_variables)

                elements = []
                for _ in range(num_variables):
                    elements.append(rng.choice(["x", "y"]))
                for _ in range(num_constants):
                    elements.append(str(rng.randint(1, 5)))

                rng.shuffle(elements)

                operators = []
                is_custom = False

                for _ in range(len(elements) - 1):
                    if custom_symbols and rng.random() < 0.2 and not is_custom:
                        is_custom = True
                        operators.append(rng.choice(custom_symbols))
                    else:
                        operators.append(rng.choices(basic_ops, weights=weights)[0])

                rng.shuffle(operators)
                operators = [""] + operators

                # Add brackets
                if rng.random() < bracket_rate:
                    position1 = rng.choice(range(len(operators)))
                    position2 = rng.choice(range(len(operators)))
                    left_position = min(position1, position2)
                    right_position = max(position1, position2)
                    elements[left_position] = "(" + elements[left_position]
                    elements[right_position] = elements[right_position] + ")"
                    if rng.random() < abs_rate:
                        elements[left_position] = "abs" + elements[left_position]

                # Build expression
                expression = ""
                for i in range(len(operators)):
                    expression += f" {operators[i]} {elements[i]}"

                if is_custom:
                    original_expression = expression
                    expression = self._simplify_mix_expression(expression, symbol_definitions)

                # Validate expression has both x and y
                has_x = 'x' in expression
                has_y = 'y' in expression

                # Test that it's not a constant (evaluate with different x,y values)
                try:
                    test_expr = expression.replace("abs(", "abs(")
                    val1 = eval(test_expr, {"x": 1, "y": 2, "abs": abs})
                    val2 = eval(test_expr, {"x": 2, "y": 3, "abs": abs})
                    is_variable = (val1 != val2)  # If values differ, it's not constant
                except:
                    is_variable = False

                if has_x and has_y and is_variable:
                    if is_custom:
                        return original_expression
                    else:
                        return expression

            except Exception:
                continue

        # Fallback: simple operation
        return "x + y"

    def _generate_expression(self, rng, symbols, max_operands):
        """Generate the main expression to evaluate"""
        num_operands = rng.randint(len(symbols), len(symbols) + 3)
        num_operands = min(num_operands, max_operands)

        operands = [rng.randint(1, 10) for _ in range(num_operands)]

        operators = symbols.copy()
        for i in range(num_operands - len(symbols) - 1):
            operators.append(rng.choice(self.base_operations + symbols))

        rng.shuffle(operators)

        components = []
        for i in range(num_operands):
            components.append(str(operands[i]))
            if i < num_operands - 1:
                components.append(operators[i])

        expression = " ".join(components)
        return expression, operands

    def _evaluate_expression(self, expression, symbol_definitions):
        """Evaluate the expression to get the answer"""
        try:
            simplified_expr = self._simplify_mix_expression(expression, symbol_definitions)
            simplified_expr = simplified_expr.replace("Abs(", "abs(")

            result = eval(simplified_expr)

            if isinstance(result, float) and result.is_integer():
                return int(result), simplified_expr
            elif isinstance(result, float):
                return round(result, 6), simplified_expr
            else:
                return result, simplified_expr

        except Exception:
            return "evaluation_error", ""

    def _simplify_mix_expression(self, expression, symbol_definitions):
        """Simplify expression with custom operators"""
        if not expression:
            return expression

        expression = expression.replace("abs(", "Abs(")

        # Tokenize
        tokens = re.findall(
            r"Abs\(|\*\*|\-?\d+\.\d+(?:[eE][\+\-]?\d+)?|\-?\d+(?:[eE][\+\-]?\d+)?|\(|\)|[a-zA-Z]+|[\+\-\*/\^%]|[^\s\w\+\-\*/\^%\(\)]+",
            expression,
        )

        # Build precedence map
        precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3, "**": 3}
        for symbol, definition in symbol_definitions.items():
            precedence[symbol] = definition["precedence"]

        operand_stack = []
        operator_stack = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if self._is_number(token) or token.isalpha():
                operand_stack.append(token)
            elif token == "Abs(" or token == "(":
                operator_stack.append(token)
            elif token == ")":
                while operator_stack and operator_stack[-1] not in ["(", "Abs("]:
                    self._process_top_operator(operand_stack, operator_stack, symbol_definitions)

                if operator_stack:
                    left_bracket = operator_stack.pop()
                    if left_bracket == "Abs(":
                        top_operand = operand_stack.pop()
                        operand_stack.append(f"Abs({top_operand})")
            else:
                while (
                    operator_stack
                    and operator_stack[-1] not in ["(", "Abs("]
                    and (
                        token not in precedence
                        or (operator_stack[-1] in precedence and precedence[operator_stack[-1]] >= precedence[token])
                    )
                ):
                    self._process_top_operator(operand_stack, operator_stack, symbol_definitions)

                operator_stack.append(token)
            i += 1

        while operator_stack:
            if operator_stack[-1] in ["(", "Abs("]:
                operator_stack.pop()
            else:
                self._process_top_operator(operand_stack, operator_stack, symbol_definitions)

        if len(operand_stack) == 1:
            return operand_stack[0]
        else:
            return ""

    def _process_top_operator(self, operand_stack, operator_stack, symbol_definitions):
        """Process top operator in the stacks"""
        operators = ["+", "-", "*", "/", "%", "^", "**"] + list(symbol_definitions.keys())

        if len(operator_stack) == 0 or len(operand_stack) < 2:
            return

        operator = operator_stack.pop()
        right_operand = operand_stack.pop()
        left_operand = operand_stack.pop()

        if self._has_operator(right_operand, operators):
            right_operand = f"({right_operand})"
        if self._has_operator(left_operand, operators):
            left_operand = f"({left_operand})"

        if operator in ["+", "-", "*", "/", "%", "^", "**"]:
            if operator in ["^", "**"]:
                operator = "**"
            result = f"{left_operand} {operator} {right_operand}"
        elif operator in symbol_definitions:
            definition = symbol_definitions[operator]

            # Check if we can evaluate condition
            if (
                self._can_eval(left_operand)
                and self._can_eval(right_operand)
                and definition["conditions"]
            ):
                left_val = eval(left_operand)
                right_val = eval(right_operand)

                if isinstance(left_val, int) and isinstance(right_val, int):
                    condition = definition["conditions"][0]["condition"]
                    if self._check_condition(left_val, right_val, condition):
                        operation = definition["conditions"][0]["operation"]
                    else:
                        operation = definition["default_operation"]
                else:
                    operation = definition["default_operation"]
            else:
                operation = definition["default_operation"]

            result = operation.replace("x", "LEFT").replace("y", "RIGHT")
            result = result.replace("LEFT", str(left_operand)).replace("RIGHT", str(right_operand))

            if self._has_custom_operator(result, symbol_definitions):
                result = self._simplify_mix_expression(result, symbol_definitions)

        result = f"({result})"
        operand_stack.append(result)

    def _is_number(self, token):
        """Check if token is a number"""
        try:
            float(token)
            return True
        except Exception:
            return False

    def _can_eval(self, operand):
        """Check if operand can be evaluated"""
        try:
            eval(operand)
            return True
        except Exception:
            return False

    def _has_operator(self, operand, operators):
        """Check if operand contains any operator"""
        for op in operators:
            if op in operand:
                return True
        return False

    def _has_custom_operator(self, operand, symbol_definitions):
        """Check if operand contains custom operator"""
        for symbol in symbol_definitions.keys():
            if symbol in operand:
                return True
        return False

    def _check_condition(self, x, y, condition):
        """Check if condition is satisfied"""
        if condition in self.condition_checks:
            return self.condition_checks[condition](x, y)
        return False

    def _format_question(self, expression, symbol_definitions, language):
        """Format the question with symbol definitions"""
        if language == "chinese":
            question = "定义 "
        else:
            question = "Define "

        for i, (symbol, definition) in enumerate(symbol_definitions.items()):
            if i > 0:
                if language == "chinese":
                    question += "和 "
                else:
                    question += "and "
            question += f"{symbol}"

            if language == "chinese":
                question += "，规则如下：\n"
            else:
                question += ", the rules are as follows:\n"

            for condition_def in definition["conditions"]:
                condition = condition_def["condition"]
                operation = condition_def["operation"]
                if language == "chinese":
                    question += f"当{condition}时，x {symbol} y = {operation}；\n"
                else:
                    question += f"when {self.condition2english[condition]}, x {symbol} y = {operation};\n"

            default_operation = definition["default_operation"]
            if len(definition["conditions"]) == 0:
                if language == "chinese":
                    question += f"实数域上，x {symbol} y = {default_operation}。\n"
                else:
                    question += f"on the real number field, x {symbol} y = {default_operation}.\n"
            else:
                if language == "chinese":
                    question += f"其他情况下，x {symbol} y = {default_operation}。\n"
                else:
                    question += f"otherwise, x {symbol} y = {default_operation}.\n"

        # Add precedence info
        if len(symbol_definitions) > 0:
            all_operators = [
                {"symbol": "**", "precedence": 3},
                {"symbol": "*", "precedence": 2},
                {"symbol": "/", "precedence": 2},
                {"symbol": "%", "precedence": 2},
                {"symbol": "+", "precedence": 1},
                {"symbol": "-", "precedence": 1},
            ]

            for symbol, definition in symbol_definitions.items():
                all_operators.append({"symbol": symbol, "precedence": definition["precedence"]})

            all_operators.sort(key=lambda x: x["precedence"], reverse=True)

            if language == "chinese":
                question += "运算优先级："
            else:
                question += "The precedence of operations: "

            current_precedence = all_operators[0]["precedence"]
            question += all_operators[0]["symbol"]

            for op in all_operators[1:]:
                if op["precedence"] == current_precedence:
                    question += " = " + op["symbol"]
                else:
                    question += " > " + op["symbol"]
                    current_precedence = op["precedence"]

            question += "。\n"

        if language == "chinese":
            question += "括号具有最高优先级，先计算括号内的表达式。\n"
        else:
            question += "Parentheses have the highest priority, and the expression inside the parentheses is calculated first.\n"

        if language == "chinese":
            question += f"请计算表达式的值: {expression}\n请将最终答案填入\\boxed{{}}中。"
        else:
            question += f"Please calculate the value of the expression: {expression}\nPlease fill your final answer in \\boxed{{}}."

        return question

    def _generate_fallback(self, seed, rng, language):
        """Generate a simple fallback problem if all attempts fail"""
        # Simple problem: a△b = a+b, evaluate 3△5
        symbol = rng.choice(self.symbols)
        a, b = rng.randint(1, 10), rng.randint(1, 10)
        answer = a + b

        if language == "chinese":
            question = f"定义 {symbol}，规则如下：\n"
            question += f"实数域上，x {symbol} y = x + y。\n"
            question += "运算优先级：** > * = / = % > + = - = {symbol}。\n"
            question += "括号具有最高优先级，先计算括号内的表达式。\n"
            question += f"请计算表达式的值: {a} {symbol} {b}\n请将最终答案填入\\boxed{{}}中。"
        else:
            question = f"Define {symbol}, the rules are as follows:\n"
            question += f"on the real number field, x {symbol} y = x + y.\n"
            question += f"The precedence of operations: ** > * = / = % > + = - = {symbol}.\n"
            question += "Parentheses have the highest priority, and the expression inside the parentheses is calculated first.\n"
            question += f"Please calculate the value of the expression: {a} {symbol} {b}\nPlease fill your final answer in \\boxed{{}}."

        return Data(
            question=question,
            answer=str(answer),
            metadata={
                "seed": seed,
                "trace_id": str(uuid.uuid4()),
                "expression": f"{a} {symbol} {b}",
                "symbol_definitions": {symbol: {"default_operation": "x + y", "conditions": [], "precedence": 1}},
                "simplified_expr": f"{a} + {b}",
                "language": language,
                "is_fallback": True,
            }
        )

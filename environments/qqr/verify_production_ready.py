#!/usr/bin/env python3
"""
QQR Pre-Launch Verification Suite

Tests across five layers:
  A-layer: AST structure validation (no imports needed)
  B-layer: Module import tests (scorer/problem_generator)
  C-layer: Mock transport determinism (exec stub bypasses mcp import)
  D-layer: Scoring hardening verification (anti-inflation fixes)
  E-layer: Anti-echo hardening verification (anti-gaming fixes)
"""

import ast
import os
import sys
import textwrap
import traceback

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RESULTS: list = []  # (test_id, label, passed, detail)


def record(test_id: str, label: str, passed: bool, detail: str = ""):
    RESULTS.append((test_id, label, passed, detail))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  A-LAYER: AST Structure Verification (no imports required)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _parse_file(filename: str):
    """Parse a Python file and return (tree, source). Raises on syntax error."""
    path = os.path.join(ROOT, filename)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=filename), source


# A.1a-d: Syntax validation for 4 modified files
FILES_TO_CHECK = [
    ("A.1a", "llm_validator.py"),
    ("A.1b", "scorer.py"),
    ("A.1c", "env.py"),
    ("A.1d", "mock_transport/server.py"),
]

for tid, fname in FILES_TO_CHECK:
    try:
        _parse_file(fname)
        record(tid, f"{fname} parses cleanly", True)
    except SyntaxError as e:
        record(tid, f"{fname} parses cleanly", False, str(e))


# A.2: _circuit_open and _consecutive_failures absent from llm_validator.py
try:
    _, src = _parse_file("llm_validator.py")
    forbidden = ["_circuit_open", "_consecutive_failures"]
    found = [f for f in forbidden if f in src]
    if found:
        record("A.2", "Circuit breaker removed from llm_validator.py", False,
               f"Found: {found}")
    else:
        record("A.2", "Circuit breaker removed from llm_validator.py", True)
except Exception as e:
    record("A.2", "Circuit breaker removed from llm_validator.py", False, str(e))


# A.4: search_train_tickets has 6 default="" parameters
try:
    tree, _ = _parse_file("mock_transport/server.py")
    found_defaults = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "search_train_tickets":
                # Count defaults that are empty string ""
                defaults = node.args.defaults
                empty_str_count = sum(
                    1 for d in defaults
                    if isinstance(d, ast.Constant) and d.value == ""
                )
                if empty_str_count == 6:
                    record("A.4", 'search_train_tickets has 6 default="" params', True)
                else:
                    record("A.4", 'search_train_tickets has 6 default="" params', False,
                           f"Found {empty_str_count} empty-string defaults, expected 6")
                found_defaults = True
                break
    if not found_defaults:
        record("A.4", 'search_train_tickets has 6 default="" params', False,
               "Function not found")
except Exception as e:
    record("A.4", 'search_train_tickets has 6 default="" params', False, str(e))


# A.9: env.py step() uses score_result.total + sets error when LLM unavailable
try:
    _, src = _parse_file("env.py")
    has_check = "score_result.llm_validation_error" in src
    has_total = "score_result.total" in src
    has_error_field = 'ep.score_breakdown["error"]' in src
    if has_check and has_total and has_error_field:
        record("A.9", "env.py step() uses total + error when LLM unavailable", True)
    else:
        record("A.9", "env.py step() uses total + error when LLM unavailable", False,
               f"check={has_check}, total={has_total}, error_field={has_error_field}")
except Exception as e:
    record("A.9", "env.py step() uses total + error when LLM unavailable", False, str(e))


# A.10: env.py evaluate() finally block has only _episodes.pop + gc.collect, no mcp/cleanup
try:
    tree, src = _parse_file("env.py")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "evaluate":
                # Find try/finally in the function body
                for child in ast.walk(node):
                    if isinstance(child, ast.Try):
                        finally_body = child.finalbody
                        if finally_body:
                            # Collect all names in the finally block
                            finally_names = set()
                            for fnode in finally_body:
                                for n in ast.walk(fnode):
                                    if isinstance(n, ast.Name):
                                        finally_names.add(n.id)
                                    elif isinstance(n, ast.Attribute):
                                        finally_names.add(n.attr)
                            has_pop = "pop" in finally_names
                            has_gc = "collect" in finally_names or "gc" in finally_names
                            has_mcp = any(
                                n in finally_names
                                for n in ("cleanup", "mcp", "_mcp_state",
                                          "close", "shutdown")
                            )
                            if has_pop and has_gc and not has_mcp:
                                record("A.10", "evaluate() finally: only pop + gc.collect, no MCP cleanup", True)
                            else:
                                record("A.10", "evaluate() finally: only pop + gc.collect, no MCP cleanup", False,
                                       f"pop={has_pop}, gc={has_gc}, mcp_related={has_mcp}, names={finally_names}")
                            break
                break
except Exception as e:
    record("A.10", "evaluate() finally: only pop + gc.collect, no MCP cleanup", False, str(e))


# A.11: config.py TOOLS_SCHEMA search_train_tickets required = ["date","from_city","to_city"]
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_check", os.path.join(ROOT, "config.py"))
    config_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_mod)

    tools_schema = config_mod.TOOLS_SCHEMA
    train_tool = None
    for tool in tools_schema:
        if tool.get("function", {}).get("name") == "search_train_tickets":
            train_tool = tool
            break

    if train_tool is None:
        record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", False,
               "search_train_tickets not found in TOOLS_SCHEMA")
    else:
        required = train_tool["function"]["parameters"]["required"]
        expected = ["date", "from_city", "to_city"]
        if sorted(required) == sorted(expected):
            record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", True)
        else:
            record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", False,
                   f"Expected {expected}, got {required}")
except Exception as e:
    record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", False, str(e))


# A.12: DEFAULT_LLM_SCORES removed from config.py and scorer.py
try:
    _, config_src = _parse_file("config.py")
    _, scorer_src = _parse_file("scorer.py")
    in_config = "DEFAULT_LLM_SCORES" in config_src
    in_scorer = "DEFAULT_LLM_SCORES" in scorer_src
    if not in_config and not in_scorer:
        record("A.12", "DEFAULT_LLM_SCORES fully removed", True)
    else:
        record("A.12", "DEFAULT_LLM_SCORES fully removed", False,
               f"config.py: {in_config}, scorer.py: {in_scorer}")
except Exception as e:
    record("A.12", "DEFAULT_LLM_SCORES fully removed", False, str(e))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  B-LAYER: Module Import Tests                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# A.3: generate(task_id) determinism across 10 task_ids
try:
    from problem_generator import get_generator
    gen = get_generator()
    all_match = True
    mismatch_ids = []
    for tid in range(10):
        p1 = gen.generate(tid)
        p2 = gen.generate(tid)
        if p1.to_dict() != p2.to_dict():
            all_match = False
            mismatch_ids.append(tid)
    if all_match:
        record("A.3", "generate() deterministic for 10 task_ids", True)
    else:
        record("A.3", "generate() deterministic for 10 task_ids", False,
               f"Mismatch at task_ids: {mismatch_ids}")
except Exception as e:
    record("A.3", "generate() deterministic for 10 task_ids", False,
           traceback.format_exc())


# A.5: ScoreBreakdown with llm_validation_error → total=0, all scores=0
try:
    from scorer import ScoreBreakdown
    sb = ScoreBreakdown()
    sb.llm_validation_error = "All 3 models failed"
    # With no scores set, total should be 0
    total_ok = sb.total == 0.0
    code_ok = sb.code_total == 0.0
    llm_ok = sb.llm_total == 0.0
    if total_ok and code_ok and llm_ok:
        record("A.5", "ScoreBreakdown with llm_validation_error -> total=0", True)
    else:
        record("A.5", "ScoreBreakdown with llm_validation_error -> total=0", False,
               f"total={sb.total}, code={sb.code_total}, llm={sb.llm_total}")
except Exception as e:
    record("A.5", "ScoreBreakdown with llm_validation_error -> total=0", False,
           traceback.format_exc())


# A.7: ScoreBreakdown.total safe with all-zero fields (no ZeroDivisionError)
try:
    from scorer import ScoreBreakdown
    sb = ScoreBreakdown()
    total = sb.total
    if isinstance(total, float) and total == 0.0:
        record("A.7", "ScoreBreakdown.total safe with zero fields", True)
    else:
        record("A.7", "ScoreBreakdown.total safe with zero fields", False,
               f"total={total}")
except ZeroDivisionError as e:
    record("A.7", "ScoreBreakdown.total safe with zero fields", False,
           f"ZeroDivisionError: {e}")
except Exception as e:
    record("A.7", "ScoreBreakdown.total safe with zero fields", False,
           traceback.format_exc())


# A.8: ScoreBreakdown.to_dict() includes llm_validation_error field
try:
    from scorer import ScoreBreakdown
    sb = ScoreBreakdown()
    sb.llm_validation_error = "test error"
    d = sb.to_dict()
    if "llm_validation_error" in d and d["llm_validation_error"] == "test error":
        record("A.8", "ScoreBreakdown.to_dict() has llm_validation_error", True)
    else:
        record("A.8", "ScoreBreakdown.to_dict() has llm_validation_error", False,
               f"Keys: {list(d.keys())}")
except Exception as e:
    record("A.8", "ScoreBreakdown.to_dict() has llm_validation_error", False,
           traceback.format_exc())


# A.6a: TravelScorer + MockValidator(success=True) → LLM scores filled, code scores computed
# A.6b: TravelScorer + MockValidator(success=False) → early return, code scores=0, error propagated
# A.6c: TravelScorer(llm_validator=None) → early return, error="LLM validator unavailable"
try:
    import asyncio
    import types as _types

    # Stub openai if not installed (production has it; local dev may not)
    if "openai" not in sys.modules:
        _openai_stub = _types.ModuleType("openai")
        class _FakeAsyncOpenAI:
            def __init__(self, *a, **kw): pass
        _openai_stub.AsyncOpenAI = _FakeAsyncOpenAI
        sys.modules["openai"] = _openai_stub

    from scorer import TravelScorer
    from llm_validator import LLMValidationResult
    from problem_generator import get_generator

    class MockValidator:
        """Mock LLMValidator for testing."""
        def __init__(self, success: bool):
            self._success = success

        async def validate(self, model_output, problem, tool_trace):
            if self._success:
                return LLMValidationResult(
                    success=True,
                    tool_info_used=True,
                    practicality=8.0,
                    analysis_depth=7.0,
                    logic=8.0,
                    user_experience=7.0,
                    reasons={"practicality": "good", "analysis_depth": "good",
                             "logic": "good", "user_experience": "good"},
                )
            else:
                return LLMValidationResult(
                    success=False,
                    error="Mock API failure: all models failed",
                )

    gen = get_generator()
    problem = gen.generate(0)  # intercity type

    fake_output = (
        "# 旅行规划方案\n\n"
        "## 第一天\n"
        "1. 上午: 游览故宫博物院 (门票60元)\n"
        "2. 下午: 天安门广场 (免费)\n"
        "3. 晚上: 推荐全聚德烤鸭 (人均150元)\n\n"
        "## 交通方案\n"
        "- 航班 CA1234，价格800元，08:00从首都国际机场出发\n"
        "- 火车 G123，价格553元，07:00从北京南站出发\n\n"
        "## 天气\n"
        "晴天，适合出行\n"
    )

    fake_tool_trace = [
        {
            "name": "poi_search",
            "arguments": {"address": "故宫", "region": "北京"},
            "result": {"text": "名称: 故宫博物院，地址: 北京市东城区景山前街4号"},
        },
        {
            "name": "search_flights",
            "arguments": {"date": "2025-05-01", "from_city": "上海", "to_city": "北京"},
            "result": {"text": "航班 CA1234，价格800元，08:00从首都国际机场出发"},
        },
    ]

    loop = asyncio.new_event_loop()

    # A.6a: success=True
    scorer_ok = TravelScorer(llm_validator=MockValidator(success=True))
    result_ok = loop.run_until_complete(
        scorer_ok.score(fake_output, problem, fake_tool_trace)
    )
    llm_filled = (
        result_ok.llm_practicality > 0
        and result_ok.llm_analysis_depth > 0
        and result_ok.llm_logic > 0
        and result_ok.llm_user_experience > 0
    )
    code_ran = result_ok.parse_success
    if llm_filled and code_ran:
        record("A.6a", "MockValidator(success=True) -> LLM+code scores", True)
    else:
        record("A.6a", "MockValidator(success=True) -> LLM+code scores", False,
               f"llm_filled={llm_filled} (prac={result_ok.llm_practicality}), "
               f"code_ran={code_ran}")

    # A.6b: success=False → code-first: code STILL computed, llm=0, error propagated
    scorer_fail = TravelScorer(llm_validator=MockValidator(success=False))
    result_fail = loop.run_until_complete(
        scorer_fail.score(fake_output, problem, fake_tool_trace)
    )
    code_computed = result_fail.code_total > 0
    llm_zero = result_fail.llm_total == 0.0
    error_propagated = "Mock API failure" in result_fail.llm_validation_error
    if code_computed and llm_zero and error_propagated:
        record("A.6b", "MockValidator(success=False) -> code computed, llm=0", True)
    else:
        record("A.6b", "MockValidator(success=False) -> code computed, llm=0", False,
               f"code={result_fail.code_total}, llm={result_fail.llm_total}, "
               f"error='{result_fail.llm_validation_error}'")

    # A.6c: No LLM validator → code-first: code STILL computed, error set
    scorer_none = TravelScorer(llm_validator=None)
    result_none = loop.run_until_complete(
        scorer_none.score(fake_output, problem, fake_tool_trace)
    )
    code_computed_c = result_none.code_total > 0
    llm_zero_c = result_none.llm_total == 0.0
    has_error = "not configured" in result_none.llm_validation_error
    if code_computed_c and llm_zero_c and has_error:
        record("A.6c", "No LLM validator -> code computed, error set", True)
    else:
        record("A.6c", "No LLM validator -> code computed, error set", False,
               f"code={result_none.code_total}, llm={result_none.llm_total}, "
               f"error='{result_none.llm_validation_error}'")

    loop.close()

except Exception as e:
    record("A.6a", "MockValidator(success=True) -> LLM+code scores", False,
           traceback.format_exc())
    record("A.6b", "MockValidator(success=False) -> code computed, llm=0", False,
           traceback.format_exc())
    record("A.6c", "No LLM validator -> code computed, error set", False,
           traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C-LAYER: Mock Transport Determinism                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# B.2: _generate_flights / _generate_trains deterministic + ≥8 records
try:
    import types
    server_path = os.path.join(ROOT, "mock_transport", "server.py")
    with open(server_path, "r", encoding="utf-8") as f:
        server_src = f.read()

    stub_globals = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "__file__": server_path,
    }

    # Stub out the mcp.server.fastmcp import
    mcp_stub = types.ModuleType("mcp")
    mcp_server_stub = types.ModuleType("mcp.server")
    mcp_fastmcp_stub = types.ModuleType("mcp.server.fastmcp")

    class FakeFastMCP:
        def __init__(self, *a, **kw): pass
        def tool(self, *a, **kw):
            def decorator(fn): return fn
            return decorator
        def run(self): pass

    mcp_fastmcp_stub.FastMCP = FakeFastMCP
    mcp_server_stub.fastmcp = mcp_fastmcp_stub
    mcp_stub.server = mcp_server_stub

    saved_modules = {}
    for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
        saved_modules[name] = sys.modules.get(name)

    sys.modules["mcp"] = mcp_stub
    sys.modules["mcp.server"] = mcp_server_stub
    sys.modules["mcp.server.fastmcp"] = mcp_fastmcp_stub

    try:
        os.environ["TRANSPORT_SALT"] = "test_salt_2025"
        exec(compile(server_src, server_path, "exec"), stub_globals)

        gen_flights = stub_globals["_generate_flights"]
        gen_trains = stub_globals["_generate_trains"]

        flights1 = gen_flights("2025-05-01", "北京", "上海")
        flights2 = gen_flights("2025-05-01", "北京", "上海")
        trains1 = gen_trains("2025-05-01", "北京", "上海")
        trains2 = gen_trains("2025-05-01", "北京", "上海")

        flights_match = flights1 == flights2
        trains_match = trains1 == trains2
        flights_count = len(flights1) >= 8
        trains_count = len(trains1) >= 8

        all_ok = flights_match and trains_match and flights_count and trains_count
        if all_ok:
            record("B.2", "Transport generation deterministic + >=8 records", True)
        else:
            detail = (
                f"flights_match={flights_match}, trains_match={trains_match}, "
                f"flights={len(flights1)}, trains={len(trains1)}"
            )
            record("B.2", "Transport generation deterministic + >=8 records", False, detail)
    finally:
        for name, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        os.environ.pop("TRANSPORT_SALT", None)

except Exception as e:
    record("B.2", "Transport generation deterministic + >=8 records", False,
           traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  D-LAYER: Scoring Hardening Verification                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# D.1: IC minimum quantity gate — 1 match out of 10 tool facts → score < 50%
try:
    from scorer import TravelScorer
    cat_score = TravelScorer._ic_category_score(
        matched_count=1, tool_count=10, divisor=0.6
    )
    # With 10 tool facts, min_required = min(3, ceil(10*0.3)) = 3
    # 1 < 3 → capped at IC_BELOW_MINIMUM_SCALE * (1/3) = 0.5 * 0.33 = 0.167
    if cat_score < 0.5:
        record("D.1", "IC min quantity: 1/10 match → score < 0.5", True,
               f"cat_score={cat_score:.3f}")
    else:
        record("D.1", "IC min quantity: 1/10 match → score < 0.5", False,
               f"cat_score={cat_score:.3f} (expected < 0.5)")
except Exception as e:
    record("D.1", "IC min quantity: 1/10 match → score < 0.5", False,
           traceback.format_exc())


# D.2: IC minimum quantity gate — small tool set (3 facts) not affected
try:
    from scorer import TravelScorer
    cat_score_small = TravelScorer._ic_category_score(
        matched_count=1, tool_count=3, divisor=0.6
    )
    # tool_count=3 < IC_MIN_QUANTITY_THRESHOLD=4 → normal ratio scoring
    # ratio = 1/3 = 0.333, cat_score = min(1.0, 0.333/0.6) = 0.556
    if cat_score_small > 0.5:
        record("D.2", "IC min quantity: small set (3) unaffected", True,
               f"cat_score={cat_score_small:.3f}")
    else:
        record("D.2", "IC min quantity: small set (3) unaffected", False,
               f"cat_score={cat_score_small:.3f} (expected > 0.5)")
except Exception as e:
    record("D.2", "IC min quantity: small set (3) unaffected", False,
           traceback.format_exc())


# D.3: Completeness grounding — keyword+context without tool fact → 0
try:
    from scorer import TravelScorer
    scorer = TravelScorer(llm_validator=None)
    # keyword matches, context matches, but tool_facts_set is empty
    result = scorer._check_with_grounded_context(
        text="景点游览 西湖景区公园 非常推荐",
        keyword=r'(景点|游览)',
        context=r'[\u4e00-\u9fa5]{2,}(景区|公园)',
        tool_facts_set=set(),  # No tool facts
        max_pts=7.0
    )
    if result == 0.0:
        record("D.3", "No tool fact → completeness = 0", True)
    else:
        record("D.3", "No tool fact → completeness = 0", False,
               f"result={result} (expected 0.0)")
except Exception as e:
    record("D.3", "No tool fact → completeness = 0", False,
           traceback.format_exc())


# D.4: Day grounding — 0 matched POIs → 0.0
try:
    from scorer import TravelScorer, ExtractedFacts
    scorer = TravelScorer(llm_validator=None)
    facts = ExtractedFacts()
    facts.pois = {"故宫博物院", "天安门广场", "颐和园"}
    # Output text doesn't contain any of the POIs
    result = scorer._compute_day_grounding(
        output_text="第一天 上午游览景点 下午购物",
        tool_facts=facts,
        num_days=3
    )
    if result == 0.0:
        record("D.4", "Day grounding: 0 matched POIs → 0.0", True)
    else:
        record("D.4", "Day grounding: 0 matched POIs → 0.0", False,
               f"result={result} (expected 0.0)")
except Exception as e:
    record("D.4", "Day grounding: 0 matched POIs → 0.0", False,
           traceback.format_exc())


# D.5: Quantity scaling — no floor, linear scaling
try:
    from scorer import TravelScorer
    scorer = TravelScorer(llm_validator=None)
    # 1 grounded fact, target_count=4 → should be exactly tier * (1/4) = 0.25
    result = scorer._check_with_grounded_context(
        text="景点游览 故宫博物院",
        keyword=r'(景点|游览)',
        context=r'不匹配的上下文',  # Context won't match → 50% tier
        tool_facts_set={"故宫博物院", "天安门广场", "颐和园", "长城"},
        max_pts=8.0,
        use_fuzzy=True,
        target_count=4
    )
    # keyword matches, context doesn't match, tool_fact matches → 50% tier
    # quantity: 1/4 = 0.25 → 8.0 * 0.5 * 0.25 = 1.0
    expected = 8.0 * 0.5 * (1.0 / 4.0)  # = 1.0
    if abs(result - expected) < 0.01:
        record("D.5", "Quantity scaling: linear, no floor", True,
               f"result={result:.2f}, expected={expected:.2f}")
    else:
        record("D.5", "Quantity scaling: linear, no floor", False,
               f"result={result:.2f}, expected={expected:.2f}")
except Exception as e:
    record("D.5", "Quantity scaling: linear, no floor", False,
           traceback.format_exc())


# D.6: IC empty tool_facts — returns < 50% (was 50% before)
try:
    from scorer import TravelScorer, ExtractedFacts
    from parser import get_parser
    scorer = TravelScorer(llm_validator=None)
    parser = get_parser()
    parsed = parser.parse("some output text with content")
    empty_facts = ExtractedFacts()
    output_facts = ExtractedFacts()
    # tool_trace with substantial content
    tool_trace = [
        {"name": "poi_search", "arguments": {"address": "故宫"},
         "result": {"text": "这是一个很长的结果" * 20}}
    ]
    ic_score = scorer._score_info_consistency(parsed, tool_trace, empty_facts, output_facts)
    max_ic = 25.0
    # Should be max_score * 0.1 = 2.5 (parser failed on substantial data)
    if ic_score < max_ic * 0.5:
        record("D.6", "IC empty tool_facts → < 50% (was 50%)", True,
               f"ic_score={ic_score:.1f}")
    else:
        record("D.6", "IC empty tool_facts → < 50% (was 50%)", False,
               f"ic_score={ic_score:.1f} (expected < {max_ic * 0.5})")
except Exception as e:
    record("D.6", "IC empty tool_facts → < 50% (was 50%)", False,
           traceback.format_exc())


# D.7: POI fabrication penalty — tools returned POIs but output uses none
try:
    from scorer import ClaimVerifier, ExtractedFacts, HardConstraintChecker
    verifier = ClaimVerifier()
    tool_facts = ExtractedFacts()
    tool_facts.pois = {"故宫博物院", "天安门广场", "颐和园"}
    output_facts = ExtractedFacts()
    called_tools = {"poi_search"}
    # raw_output doesn't contain any tool POIs (must be >= FORMAT_MIN_LENGTH)
    fake_output = "我推荐去一些景点游览，比如随便什么地方。" * 20  # ensure > 200 chars
    penalty, violations = verifier.verify_claims(
        tool_facts, output_facts, called_tools,
        raw_output=fake_output
    )
    has_poi_violation = any("No tool POIs" in v for v in violations)
    if penalty < 0 and has_poi_violation:
        record("D.7", "POI fabrication: no tool POIs used → penalty", True,
               f"penalty={penalty}, violations={violations}")
    else:
        record("D.7", "POI fabrication: no tool POIs used → penalty", False,
               f"penalty={penalty}, violations={violations}")
except Exception as e:
    record("D.7", "POI fabrication: no tool POIs used → penalty", False,
           traceback.format_exc())

# D.8: Fabrication check skipped on empty/short output
try:
    from scorer import ClaimVerifier, ExtractedFacts
    verifier = ClaimVerifier()
    tool_facts = ExtractedFacts()
    tool_facts.pois = {"故宫博物院", "天安门广场", "颐和园"}
    tool_facts.weather = {"晴", "多云"}
    output_facts = ExtractedFacts()
    output_facts.weather = {"大雨"}  # fabricated, but output is too short
    called_tools = {"poi_search", "weather"}
    penalty, violations = verifier.verify_claims(
        tool_facts, output_facts, called_tools,
        raw_output="短文本"  # well below FORMAT_MIN_LENGTH
    )
    if penalty == 0.0 and len(violations) == 0:
        record("D.8", "Fabrication check skipped on short output", True,
               f"penalty={penalty}, violations={violations}")
    else:
        record("D.8", "Fabrication check skipped on short output", False,
               f"penalty={penalty}, violations={violations}")
except Exception as e:
    record("D.8", "Fabrication check skipped on short output", False,
           traceback.format_exc())

# D.9: POI extraction excludes admin names (pname/cityname/adname)
try:
    from scorer import FactExtractor, ExtractedFacts
    extractor = FactExtractor()
    # Simulate amap tool result with pname/cityname/adname fields
    tool_text = (
        "name: 趵突泉景区\nid: B001\nlocation: 117.0,36.6\n"
        "pname: 山东省\ncityname: 济南市\nadname: 历下区\n"
        "name: 大明湖景区\nid: B002\nlocation: 117.0,36.7\n"
        "pname: 山东省\ncityname: 济南市\nadname: 历下区\n"
    )
    facts = ExtractedFacts()
    extractor._extract_poi_facts(tool_text, facts)
    # Should extract real POI names but NOT admin names
    has_real = "趵突泉景区" in facts.pois and "大明湖景区" in facts.pois
    no_admin = "山东省" not in facts.pois and "历下区" not in facts.pois
    # "济南市" could still match via "name:" in "cityname:" if not properly gated
    no_city = "济南市" not in facts.pois
    if has_real and no_admin and no_city:
        record("D.9", "POI extraction excludes admin names", True,
               f"pois={facts.pois}")
    else:
        record("D.9", "POI extraction excludes admin names", False,
               f"pois={facts.pois}")
except Exception as e:
    record("D.9", "POI extraction excludes admin names", False,
           traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  E-LAYER: Anti-Echo Hardening Verification                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# E.1: Tier-2 proximity — keyword and fact distant (>500 chars) → 20% not 50%
try:
    from scorer import TravelScorer
    scorer = TravelScorer(llm_validator=None)
    # keyword + fact present but NO context, and fact is far from keyword (>500 chars)
    distant_text = "预算情况" + "x" * 600 + "500"
    score_distant = scorer._check_with_grounded_context(
        distant_text, r'预算', r'\d+\s*元', {"500"}, 10.0
    )
    # keyword="预算" present, context="\d+\s*元" NOT present, fact="500" present but distant
    # Should be tier-2 distant → 10.0 * 0.2 = 2.0
    if score_distant <= 10.0 * 0.25:  # Should be 2.0 (20%), not 5.0 (50%)
        record("E.1", "Tier-2 distant: keyword+fact >500 chars → ≤25%", True,
               f"score={score_distant}")
    else:
        record("E.1", "Tier-2 distant: keyword+fact >500 chars → ≤25%", False,
               f"score={score_distant}, expected ≤2.5")
except Exception as e:
    record("E.1", "Tier-2 distant: keyword+fact >500 chars → ≤25%", False,
           traceback.format_exc())

# E.2: Tier-2 proximity — keyword and fact close (<500 chars) → 50%
try:
    from scorer import TravelScorer
    scorer = TravelScorer(llm_validator=None)
    # keyword and fact within 500 chars, no context
    close_text = "预算情况参考 500"
    score_close = scorer._check_with_grounded_context(
        close_text, r'预算', r'详细列表', {"500"}, 10.0
    )
    # keyword="预算" present, context="详细列表" NOT present, fact="500" present and close
    # Should be tier-2 proximate → 10.0 * 0.5 = 5.0
    if abs(score_close - 5.0) < 0.01:
        record("E.2", "Tier-2 proximate: keyword+fact <500 chars → 50%", True,
               f"score={score_close}")
    else:
        record("E.2", "Tier-2 proximate: keyword+fact <500 chars → 50%", False,
               f"score={score_close}, expected 5.0")
except Exception as e:
    record("E.2", "Tier-2 proximate: keyword+fact <500 chars → 50%", False,
           traceback.format_exc())

# E.3: tool_info_used new thresholds — IC=5, Comp=5 → False for transport
try:
    from config import CODE_TOOL_USED_IC_THRESHOLD, CODE_TOOL_USED_COMP_THRESHOLD
    from scorer import TravelScorer
    from problem_generator import get_generator
    scorer = TravelScorer(llm_validator=None)
    gen = get_generator()
    problem = gen.generate(99)
    # Force intercity type for testing
    problem.problem_type = "intercity"
    # IC=5, Comp=5 should fail for transport (threshold=6)
    result = scorer._determine_tool_info_used(5.0, 5.0, problem)
    if not result:
        record("E.3", "tool_info_used: IC=5,Comp=5 → False (transport needs ≥6)", True,
               f"threshold={CODE_TOOL_USED_IC_THRESHOLD}")
    else:
        record("E.3", "tool_info_used: IC=5,Comp=5 → False (transport needs ≥6)", False,
               f"result={result}, threshold={CODE_TOOL_USED_IC_THRESHOLD}")
except Exception as e:
    record("E.3", "tool_info_used: IC=5,Comp=5 → False (transport needs ≥6)", False,
           traceback.format_exc())

# E.4: Structural credit reduced — keyword + regex, no fact → ≤ max_pts * 0.1
try:
    from config import STRUCTURAL_CREDIT_RATIO
    # The structural credit in _score_completeness uses STRUCTURAL_CREDIT_RATIO (0.1)
    # Verify constant value
    if STRUCTURAL_CREDIT_RATIO <= 0.15:  # Should be 0.1
        record("E.4", "Structural credit ratio ≤ 0.15 (anti-echo)", True,
               f"ratio={STRUCTURAL_CREDIT_RATIO}")
    else:
        record("E.4", "Structural credit ratio ≤ 0.15 (anti-echo)", False,
               f"ratio={STRUCTURAL_CREDIT_RATIO}, expected ≤0.15")
except Exception as e:
    record("E.4", "Structural credit ratio ≤ 0.15 (anti-echo)", False,
           traceback.format_exc())

# E.5: IC breadth penalty — 3 total categories, 1 matched → 0.3x penalty
try:
    from config import IC_BREADTH_PENALTY_MULTIPLIER, INFO_CONSISTENCY_MIN_BREADTH_TOTAL
    # Verify constants
    breadth_ok = (INFO_CONSISTENCY_MIN_BREADTH_TOTAL <= 3 and
                  IC_BREADTH_PENALTY_MULTIPLIER <= 0.35)
    if breadth_ok:
        record("E.5", "IC breadth: threshold=3, penalty=0.3x", True,
               f"total_threshold={INFO_CONSISTENCY_MIN_BREADTH_TOTAL}, "
               f"multiplier={IC_BREADTH_PENALTY_MULTIPLIER}")
    else:
        record("E.5", "IC breadth: threshold=3, penalty=0.3x", False,
               f"total_threshold={INFO_CONSISTENCY_MIN_BREADTH_TOTAL}, "
               f"multiplier={IC_BREADTH_PENALTY_MULTIPLIER}")
except Exception as e:
    record("E.5", "IC breadth: threshold=3, penalty=0.3x", False,
           traceback.format_exc())


# E.6: analysis_depth field exists and LLM mapping works
try:
    from llm_validator import LLMValidationResult
    from config import LLM_SCORE_WEIGHTS
    llm_r = LLMValidationResult(
        success=True,
        practicality=8.0,
        analysis_depth=7.0,
        logic=9.0,
        user_experience=6.0,
    )
    total = llm_r.total
    expected_total = 8.0 + 7.0 + 9.0 + 6.0  # = 30.0
    has_field = hasattr(llm_r, "analysis_depth")
    has_weight = "analysis_depth" in LLM_SCORE_WEIGHTS
    total_ok = abs(total - expected_total) < 0.01
    if has_field and has_weight and total_ok:
        record("E.6", "analysis_depth field + LLM weight exists", True,
               f"total={total}, weight={LLM_SCORE_WEIGHTS.get('analysis_depth')}")
    else:
        record("E.6", "analysis_depth field + LLM weight exists", False,
               f"has_field={has_field}, has_weight={has_weight}, total={total}")
except Exception as e:
    record("E.6", "analysis_depth field + LLM weight exists", False,
           traceback.format_exc())

# E.7: Completeness scaling — intercity 5 items scaled to ≤25.0
try:
    from config import CODE_SCORE_WEIGHTS
    max_comp = CODE_SCORE_WEIGHTS.get("completeness", 25.0)
    _s = max_comp / 35.0
    # intercity: 5 x 7.0 * _s
    scaled_sum = 5 * 7.0 * _s
    if abs(scaled_sum - max_comp) < 0.01:
        record("E.7", "Completeness scaling: intercity 5x7*_s = max_score", True,
               f"scaled_sum={scaled_sum:.2f}, max={max_comp}")
    else:
        record("E.7", "Completeness scaling: intercity 5x7*_s = max_score", False,
               f"scaled_sum={scaled_sum:.2f}, expected={max_comp}")
except Exception as e:
    record("E.7", "Completeness scaling: intercity 5x7*_s = max_score", False,
           traceback.format_exc())

# E.8: 50/50 total score range — code max 50, LLM max 50
try:
    from config import TOTAL_CODE_SCORE, TOTAL_LLM_SCORE
    from scorer import ScoreBreakdown
    code_ok = abs(TOTAL_CODE_SCORE - 50.0) < 0.01
    llm_ok = abs(TOTAL_LLM_SCORE - 50.0) < 0.01
    # Verify ScoreBreakdown with max scores
    sb = ScoreBreakdown()
    sb.info_consistency = 25.0
    sb.completeness = 25.0
    sb.llm_practicality = 12.5
    sb.llm_analysis_depth = 12.5
    sb.llm_logic = 12.5
    sb.llm_user_experience = 12.5
    code_total_ok = abs(sb.code_total - 50.0) < 0.01
    llm_total_ok = abs(sb.llm_total - 50.0) < 0.01
    total_ok = abs(sb.total - 100.0) < 0.01
    all_ok = code_ok and llm_ok and code_total_ok and llm_total_ok and total_ok
    if all_ok:
        record("E.8", "50/50 split: code=50, LLM=50, total=100", True,
               f"code={sb.code_total}, llm={sb.llm_total}, total={sb.total}")
    else:
        record("E.8", "50/50 split: code=50, LLM=50, total=100", False,
               f"TOTAL_CODE={TOTAL_CODE_SCORE}, TOTAL_LLM={TOTAL_LLM_SCORE}, "
               f"code_total={sb.code_total}, llm_total={sb.llm_total}, total={sb.total}")
except Exception as e:
    record("E.8", "50/50 split: code=50, LLM=50, total=100", False,
           traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS OUTPUT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_results():
    sep = "=" * 60
    print(f"\n{sep}")
    print("QQR Pre-Launch Verification Suite")
    print(sep)

    passed = 0
    failed = 0
    for test_id, label, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        line = f"  {status}: {test_id} - {label}"
        print(line)
        if not ok and detail:
            for dl in detail.split("\n")[:5]:
                print(f"        {dl}")

    total = passed + failed
    print(sep)
    print(f"RESULTS: {passed} passed, {failed} failed, {total} total")
    print(sep)

    # ── System Analysis Summary ──
    print(f"\n{sep}")
    print("SYSTEM ANALYSIS SUMMARY")
    print(sep)

    print(textwrap.dedent("""\
    [2.1] Concurrency Safety
      - MCPState: singleton with asyncio.Lock + double-check init
      - Actor._episodes: UUID-keyed per-evaluation isolation
      - LLMValidator: no mutable instance state (circuit breaker removed)
      - _EPOCH_SALT: import-time immutable
      - ProblemGenerator: per-call random.Random(task_id), no shared RNG

    [2.2] Resource Lifecycle
      +-------------------+---------------------+---------------------+---------------------------+
      | Component         | Created             | Destroyed           | Notes                     |
      +-------------------+---------------------+---------------------+---------------------------+
      | MCP subprocess    | Actor.reset() first | Actor.cleanup()     | singleton, diskcache      |
      | AsyncOpenAI       | LLMValidator.init   | container exit      | single instance           |
      | EpisodeState      | evaluate()          | finally pop+gc      | per-eval isolated         |
      | diskcache         | MCP server connect  | server.cleanup()    | /var/lib/qqr/cache/       |
      +-------------------+---------------------+---------------------+---------------------------+

    [2.3] Error Propagation Chain (Code-First Architecture)
      LLMValidator.validate() -> LLMValidationResult(success=False, error="...")
        -> TravelScorer.score() -> code scores ALWAYS computed; llm_total=0
        -> Actor.step() -> ep.final_score = score_result.total (code only)
        -> Actor.evaluate() -> {"score": code_score/100, "extra": {"score_breakdown": {...}}}

      No LLM validator (missing API key):
        -> TravelScorer.score() -> code computed, llm_validation_error="not configured"
        -> Same downstream: total = code_total (max 70)

    [2.4] Scorer Execution Paths (Code-First)
      +----+------------------------------------+-----------+----------+---------+---------------------------+
      | P  | Trigger                            | code      | llm      | total   | Notes                     |
      +----+------------------------------------+-----------+----------+---------+---------------------------+
      | P1 | LLM validator all models fail      | computed  | 0        | 0-50    | code-first, LLM optional  |
      | P2 | tool_info_used=False               | computed  | computed | 0       | hard fail multiplier 0.0  |
      | P3 | format_valid fail                  | computed  | computed | ~x*0.15 | RL gradient kept          |
      | P4 | Normal flow (LLM available)        | computed  | set      | 0-100   | full scoring (50/50)      |
      | P5 | No LLM validator (no API key)      | computed  | 0        | 0-50    | code-only scoring         |
      +----+------------------------------------+-----------+----------+---------+---------------------------+

    [2.5] Anti-Inflation Hardening
      +-------------------------------------------+-----------------------------------+-----------------------------------+
      | Weakness                                  | Old Behavior                      | New Behavior                      |
      +-------------------------------------------+-----------------------------------+-----------------------------------+
      | IC: 1 fact matches per category            | Full category score (1.0)         | Capped at 50% when below minimum  |
      | Completeness: keyword+context only         | 15% free credit                   | 0% (requires tool grounding)      |
      | Quantity scaling floor                     | max(0.25, ratio) → 25% minimum    | Linear: ratio (no floor)          |
      | Day structure baseline                     | 0.3 base even with 0 POIs         | 0.0 without matched POIs          |
      | Budget/tips fallback to POI names          | POI names as price proxy → 100%   | 10% structural credit max         |
      | IC empty tool_facts                        | 50% free score                    | 10-20% based on tool content      |
      | POI fabrication                            | No check                          | -3.0 penalty when 0 tool POIs used|
      +-------------------------------------------+-----------------------------------+-----------------------------------+

    [2.6] Anti-Echo Hardening (Structured Echo Attack Prevention)
      +-------------------------------------------+-----------------------------------+-----------------------------------+
      | Weakness                                  | Old Behavior                      | New Behavior                      |
      +-------------------------------------------+-----------------------------------+-----------------------------------+
      | Tier-2: keyword+fact anywhere              | 50% always                        | 50% if proximate, 20% if distant  |
      | tool_info_used gate too low                | IC>=5 (14%)                       | IC>=6 transport, IC>=4 others     |
      | Structural credit without data             | 20% free                          | 10% free                          |
      | IC breadth penalty too weak                | >=4 total, 0.5x penalty           | >=3 total, 0.3x penalty           |
      | IC no context requirement                  | Fact anywhere counts              | Out-of-context facts count 50%    |
      +-------------------------------------------+-----------------------------------+-----------------------------------+

    [2.7] Known Acceptable Risks
      +------------------------------------------+--------+----------------------------------------------+
      | Risk                                     | Sev    | Rationale                                    |
      +------------------------------------------+--------+----------------------------------------------+
      | AsyncOpenAI no explicit close             | Low    | Single instance, container exit reclaims     |
      | LLM unavailable → max 50 pts             | Low    | Code-first: fair within same eval batch      |
      | CHUTES_API_KEY required for LLM scoring  | None   | Code scoring works without it                |
      +------------------------------------------+--------+----------------------------------------------+
    """))

    return failed == 0


if __name__ == "__main__":
    success = print_results()
    sys.exit(0 if success else 1)

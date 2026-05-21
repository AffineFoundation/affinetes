"""
QQR-based Travel Planning Evaluation Environment - Actor Implementation.

Fully reuses QQR's MCP tool system for real tool invocation.

Core reuse:
- qqr.rollout.agent_rollout.MCPState - MCP server state management (singleton)
- qqr.mcp.MCPServerStdioCacheable - Cacheable MCP server
- qqr.tools.amap - AMap tools server
- qqr.tools.mock_transport - Mock transport tools server

Scoring structure:
- Hard constraints: format, tool info usage, required tools called, tool quality gate
- Code score (70 pts): info consistency(35), completeness(35) with grounding verification
- LLM score (30 pts): smooth coupling with code score via LLM_CODE_RATIO_FACTOR
"""

import asyncio
import gc
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List


logger = logging.getLogger(__name__)

# ==================== Reuse QQR's MCP System ====================
# Import from mcp_wrapper (copied from QQR to avoid slime dependency)
from mcp_wrapper import MCPServerStdioCacheable, MCPServerStdioParams, MCPState
from affent_runner import run_affent_agent

# Import from local modules
from config import (
    SYSTEM_PROMPT,
    MAX_TOOL_STEPS,
    AMAP_MAPS_API_KEY,
    PYTHONPATH,
    REQUIRES_TRANSPORT,
)
from problem_generator import TravelProblem, get_generator
from scorer import TravelScorer
from llm_validator import get_llm_evaluator

from affinetes.core.openenv import OpenEnvResponse

# ==================== MCP Server Configuration (QQR Format) ====================
# Fix epoch salt at import time — stable within a single process lifetime
# Daily rotation + per-process jitter for anti-memorization (P4)
_DAILY_SALT = str(int(time.time()) // 86400)
_PROCESS_SALT = uuid.uuid4().hex[:8]
_EPOCH_SALT = os.getenv("TRANSPORT_SALT", f"{_DAILY_SALT}_{_PROCESS_SALT}")

# Cache size from env var (default 512MB, replaces maxsize * 4KB estimate)
try:
    _CACHE_SIZE_MB = int(os.getenv("QQR_CACHE_SIZE_MB", "4096"))
except (ValueError, TypeError):
    _CACHE_SIZE_MB = 4096


_AMAP_CACHE_TTL = 2592000   # 30 days — AMap data (routes, POIs, weather) is stable
_TRANSPORT_CACHE_TTL = 172800  # 48 hours — transport data rotates with epoch salt


def _round_coord(coord_str: str, decimals: int = 3) -> str:
    """Round coordinate string to N decimal places."""
    coord_str = coord_str.replace('\uff0c', ',').replace(' ', '')
    parts = coord_str.split(',')
    if len(parts) == 2:
        try:
            return f"{round(float(parts[0]), decimals)},{round(float(parts[1]), decimals)}"
        except (ValueError, OverflowError):
            pass
    return coord_str


def _bucket_radius(radius) -> int:
    """Bucket around_search radius into standard tiers for cache key."""
    try:
        r = int(radius)
    except (ValueError, TypeError):
        return 3000
    if r <= 1500:
        return 1000
    if r <= 4000:
        return 3000
    return 5000


def _normalize_city(name: str) -> str:
    """Normalize city name: '杭州市' → '杭州', '乌鲁木齐市' → '乌鲁木齐'."""
    if len(name) > 2 and name[-1] in ("市", "县", "区"):
        return name[:-1]
    return name


def _normalize_amap_args(tool_name: str, arguments: dict) -> dict:
    """Normalize AMap tool arguments for better cache hit rate.

    - direction: round coordinates to 3 decimal places (~111m)
    - around_search: round coordinates to 3 decimal places + bucket radius
    - weather/poi_search/around_search: normalize city names ('杭州市' → '杭州')
    - All tools: strip whitespace from string arguments
    """
    if not arguments:
        return arguments

    args = dict(arguments)  # Shallow copy — tool args are flat strings

    if tool_name == "direction":
        for coord_key in ("origin", "destination"):
            if coord_key in args and isinstance(args[coord_key], str):
                args[coord_key] = _round_coord(args[coord_key], 3)
        if "waypoints" in args and isinstance(args["waypoints"], str) and args["waypoints"]:
            parts = args["waypoints"].split(";")
            args["waypoints"] = ";".join(_round_coord(p, 3) for p in parts)

    elif tool_name == "around_search":
        if "location" in args and isinstance(args["location"], str):
            args["location"] = _round_coord(args["location"], 3)
        if "radius" in args:
            args["radius"] = _bucket_radius(args["radius"])

    # Normalize city name fields
    for city_key in ("city", "region"):
        if city_key in args and isinstance(args[city_key], str):
            args[city_key] = _normalize_city(args[city_key].strip())

    # Strip whitespace from all string values
    for k, v in args.items():
        if isinstance(v, str):
            args[k] = v.strip()

    return args


def mcp_server_config_fn() -> list:
    """
    MCP server configuration function.

    Returns QQR-format MCP server list with AMap and Transport only.
    API keys are passed via environment variables.
    """
    # AMap server configuration
    amap_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "amap_server"],
        env={
            "AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY or "",
            "PYTHONPATH": PYTHONPATH or "",
        },
    )

    amap_server = MCPServerStdioCacheable(
        name="AMap",
        params=amap_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=_AMAP_CACHE_TTL,
        cache_maxsize=32768,
        cache_size_limit_mb=_CACHE_SIZE_MB,
        concurrency_limit=16,
        key_normalizer=_normalize_amap_args,
    )

    # Transport server configuration
    # Uses deterministic algorithmic generation (no LLM dependency)
    # TRANSPORT_SALT changes weekly for anti-memorization
    epoch_salt = _EPOCH_SALT
    transport_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "mock_transport.server"],
        env={
            "PYTHONPATH": PYTHONPATH or "",
            "TRANSPORT_SALT": epoch_salt,
        },
    )
    transport_server = MCPServerStdioCacheable(
        name="Transport",
        params=transport_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=120,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=_TRANSPORT_CACHE_TTL,
        cache_maxsize=32768,
        cache_size_limit_mb=_CACHE_SIZE_MB,
        concurrency_limit=4,
        cache_key_salt=epoch_salt,  # Include salt so cache invalidates on salt rotation
    )

    return [amap_server, transport_server]


# ==================== Step Reward Calculator ====================
class StepRewardCalculator:
    """Calculate intermediate rewards after each tool call step."""

    @staticmethod
    def calculate_step_reward(
        tool_name: str,
        tool_args: Dict,
        tool_result: str,
        problem: TravelProblem,
        step_number: int,
        called_tools_so_far: set,
    ) -> float:
        """
        Calculate reward for a single tool call step.

        Returns a float in [0, 1]:
          0.4 * tool_selection_score
          + 0.3 * arg_quality_score
          + 0.3 * result_usefulness_score
        """
        tool_sel = StepRewardCalculator._tool_selection_score(
            tool_name, problem, step_number, called_tools_so_far
        )
        arg_qual = StepRewardCalculator._arg_quality_score(tool_name, tool_args)
        result_use = StepRewardCalculator._result_usefulness_score(tool_result)

        return round(0.4 * tool_sel + 0.3 * arg_qual + 0.3 * result_use, 4)

    @staticmethod
    def _tool_selection_score(
        tool_name: str,
        problem: TravelProblem,
        step_number: int,
        called_tools_so_far: set,
    ) -> float:
        """Is this the right tool to call at this step?"""
        required = set(problem.required_tools)
        score = 0.0

        # Calling a required tool earns base score
        if tool_name in required:
            score += 0.6
        else:
            score += 0.2  # Non-required but possibly useful

        # Bonus for not repeating the same tool
        if tool_name not in called_tools_so_far:
            score += 0.2

        # Bonus for calling transport tools when problem requires transport
        if problem.problem_type in REQUIRES_TRANSPORT:
            if tool_name in {"search_flights", "search_train_tickets"}:
                score += 0.2

        # Diversity penalty: penalize repeating a tool to discourage
        # redundant calls (e.g., 12x poi_search instead of direction/around)
        if step_number > 3 and tool_name in called_tools_so_far:
            uncalled_required = required - called_tools_so_far - {tool_name}
            if uncalled_required:
                # Strong penalty: required tools still uncalled
                base_penalty = 0.30
                escalation = 0.05 * max(0, step_number - 5)
                score -= min(0.50, base_penalty + escalation)
            elif step_number > 6:
                # Diminishing returns: all required tools called but still
                # repeating at late stage — mild escalating penalty
                score -= min(0.20, 0.05 * (step_number - 6))

        return max(0.0, min(1.0, score))

    @staticmethod
    def _arg_quality_score(tool_name: str, args: Dict) -> float:
        """Are the arguments well-formed?"""
        if not args:
            return 0.0

        if tool_name == "poi_search":
            address = args.get("address", "")
            if address and len(address) >= 2:
                return 1.0 if re.search(r'[\u4e00-\u9fa5]', address) else 0.5
            return 0.2

        elif tool_name == "direction":
            origin = args.get("origin", "")
            dest = args.get("destination", "")
            coord_re = re.compile(r'^\d+\.\d+\s*,\s*\d+\.\d+$')
            origin_ok = bool(coord_re.match(origin.strip())) if origin else False
            dest_ok = bool(coord_re.match(dest.strip())) if dest else False
            if origin_ok and dest_ok:
                return 1.0
            elif origin_ok or dest_ok:
                return 0.5
            return 0.2

        elif tool_name == "around_search":
            loc = args.get("location", "")
            coord_re = re.compile(r'^\d+\.\d+\s*,\s*\d+\.\d+$')
            return 1.0 if (loc and coord_re.match(loc.strip())) else 0.3

        elif tool_name == "weather":
            city = args.get("city", "")
            return 1.0 if (city and re.match(r'^[\u4e00-\u9fa5]{2,10}$', city)) else 0.3

        elif tool_name in ("search_flights", "search_train_tickets"):
            date = args.get("date", "")
            from_c = args.get("from_city", "")
            to_c = args.get("to_city", "")
            date_ok = bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date))
            cities_ok = bool(from_c and to_c)
            if date_ok and cities_ok:
                return 1.0
            elif date_ok or cities_ok:
                return 0.5
            return 0.2

        return 0.5  # Unknown tool, neutral

    @staticmethod
    def _result_usefulness_score(result: str) -> float:
        """Is the tool result useful (non-empty, non-error)?"""
        if not result or len(result.strip()) < 10:
            return 0.0

        # Check for error patterns
        result_lower = result.lower()
        if len(result) < 100:
            error_keywords = ["error", "failed", "失败", "错误", "无效", "无法"]
            if any(kw in result_lower for kw in error_keywords):
                return 0.1

        # Longer results with content are more useful
        if len(result) > 500:
            return 1.0
        elif len(result) > 100:
            return 0.8
        return 0.5


# Per-episode tool call budget based on what a quality travel plan needs:
# - A good plan covers 3-5 POIs, 3-5 routes, 2-3 nearby searches, 1-2 weather
# - Total ~15 tool calls, matching MAX_TOOL_STEPS=15
# - Excess calls are wasteful (repeating same POI, trying all 4 direction modes)
# - Models exceeding budget get a text hint to use existing data
_TOOL_CALL_BUDGET = {
    "direction": 10,      # Need ~5, budget 10
    "poi_search": 10,     # Need ~5, budget 10
    "around_search": 6,   # Need ~3, budget 6
    "weather": 3,         # Need 1-2, budget 3
}


@dataclass
class EpisodeState:
    """Episode state container."""
    episode_id: str
    task_id: int
    seed: int
    problem: TravelProblem
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)
    current_step: int = 0
    done: bool = False
    final_score: float = 0.0
    score_breakdown: Optional[Dict] = None
    _score_breakdown_full: Optional[Dict] = None  # Internal debug only
    tool_call_counts: Dict[str, int] = field(default_factory=dict)  # per-tool call counter


class Actor:
    """
    QQR-based travel planning evaluation environment.

    Fully reuses QQR's MCPState to manage and invoke MCP tools.
    """

    def __init__(
        self,
        enable_llm_validator: bool = True,
        llm_validator_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
    ):
        self._episodes: Dict[str, EpisodeState] = {}
        self._generator = get_generator()
        self._llm_validator_model = llm_validator_model

        # ==================== Reuse QQR's MCPState ====================
        # MCPState is singleton, initialized with our config function
        self._mcp_state = MCPState(mcp_server_config_fn)
        self._mcp_initialized = False

        # ==================== Production Hardening ====================
        # Protect lazy initialization of _llm_validator from concurrent access
        self._init_lock = asyncio.Lock()
        # Limit max concurrent evaluations to prevent resource exhaustion
        self._eval_semaphore = asyncio.Semaphore(20)
        # Periodic GC: trigger gc.collect() every _gc_interval evaluations
        self._eval_count = 0
        self._gc_interval = 20

        # LLM Evaluator with scorer groups (quality + grounding)
        self._llm_evaluator = None
        if enable_llm_validator:
            self._llm_evaluator = get_llm_evaluator()

        self._scorer = TravelScorer(llm_validator=self._llm_evaluator)

    async def _ensure_mcp_initialized(self):
        """Ensure MCP servers are initialized."""
        if not self._mcp_initialized:
            await self._mcp_state.get_mcp_servers()
            self._mcp_initialized = True

    async def reset(
        self,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> OpenEnvResponse:
        """Reset environment and generate new problem."""
        # Ensure MCP servers are initialized
        await self._ensure_mcp_initialized()

        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        task_id = task_id if task_id is not None else (seed & 0x7FFFFFFF)

        problem = self._generator.generate(task_id)
        prompt = self._generator.to_prompt(problem)

        episode_id = uuid.uuid4().hex
        ep = EpisodeState(
            episode_id=episode_id,
            task_id=task_id,
            seed=seed,
            problem=problem,
            conversation=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        self._episodes[episode_id] = ep

        return OpenEnvResponse(
            observation=prompt,
            reward=0.0,
            done=False,
            episode_id=episode_id,
            info={
                "task_id": task_id,
                "seed": seed,
                "problem_type": problem.problem_type,
                "problem": problem.to_dict(),
                "available_tools": [t["function"]["name"] for t in self._mcp_state.tools],
            },
        )

    async def step(
        self,
        action: str,
        episode_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
    ) -> OpenEnvResponse:
        """
        Execute one step.

        Args:
            action: Model output text content
            episode_id: Episode ID
            tool_calls: OpenAI format tool call list
        """
        ep = self._episodes.get(episode_id)
        if ep is None or ep.done:
            return OpenEnvResponse(
                observation="Error: Invalid episode",
                reward=0.0,
                done=True,
                episode_id=episode_id
            )

        ep.current_step += 1

        # If tool calls exist, execute using MCPState
        logger.debug("step: tool_calls=%d, current_step=%d, MAX=%d", len(tool_calls) if tool_calls else 0, ep.current_step, MAX_TOOL_STEPS)
        if tool_calls and ep.current_step <= MAX_TOOL_STEPS:
            # Normalize: ensure every tool_call has a stable id
            for tc in tool_calls:
                if not tc.get("id"):
                    tc["id"] = f"call_{uuid.uuid4().hex[:24]}"

            # Standard OpenAI assistant message with tool_calls
            tc_list = []
            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                # Ensure arguments is a JSON string (some models return dict)
                if not isinstance(raw_args, str):
                    raw_args = json.dumps(raw_args, ensure_ascii=False)
                tc_list.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": func.get("name", "unknown"),
                        "arguments": raw_args,
                    },
                })
            assistant_msg = {
                "role": "assistant",
                "tool_calls": tc_list,
            }
            # OpenAI spec: content is optional when tool_calls present
            if action:
                assistant_msg["content"] = action
            ep.conversation.append(assistant_msg)
            logger.debug("Assistant message with %d tool_calls", len(tc_list))
        else:
            # No tool calls, add assistant message directly
            ep.conversation.append({"role": "assistant", "content": action or ""})

        if tool_calls and ep.current_step <= MAX_TOOL_STEPS:
            tool_results = []

            for call in tool_calls:
                func = call.get("function", {})
                name = func.get("name", "")
                call_id = call["id"]  # Use the normalized id from above

                # Parse arguments
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}

                # ==================== Use QQR's MCPState to call tools ====================
                # Per-episode tool call budget check
                ep.tool_call_counts[name] = ep.tool_call_counts.get(name, 0) + 1
                budget = _TOOL_CALL_BUDGET.get(name)
                if budget and ep.tool_call_counts[name] > budget:
                    result_content = (
                        f"Tool call budget exceeded: {name} called "
                        f"{ep.tool_call_counts[name]} times (limit: {budget}). "
                        f"Use the data from previous calls."
                    )
                else:
                    tool_call_dict = {
                        "id": call_id,
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }

                    # Call MCPState.call_tool (with timeout to prevent hung subprocesses)
                    try:
                        tool_response = await asyncio.wait_for(
                            self._mcp_state.call_tool(tool_call_dict),
                            timeout=120,
                        )
                        result_content = tool_response.get("content", "")
                    except asyncio.TimeoutError:
                        logger.error("Tool call %s timed out after 120s", name)
                        result_content = f"tool_call_timeout: {name} did not respond within 120s"
                    except Exception as e:
                        logger.error("Tool call %s failed: %s", name, e)
                        result_content = f"tool_call_failed: {str(e)[:200]}"

                # Record tool call
                ep.tool_trace.append({
                    "name": name,
                    "arguments": args,
                    "result": {"text": result_content},
                })

                # Calculate step reward
                called_tools_so_far = set(t["name"] for t in ep.tool_trace[:-1])
                step_reward = StepRewardCalculator.calculate_step_reward(
                    tool_name=name,
                    tool_args=args,
                    tool_result=result_content,
                    problem=ep.problem,
                    step_number=ep.current_step,
                    called_tools_so_far=called_tools_so_far,
                )
                ep.step_rewards.append(step_reward)

                tool_results.append({
                    "tool": name,
                    "call_id": call_id,
                    "result": self._truncate_tool_result(result_content, max_chars=3000),
                })

                # Check for env errors after EACH tool call — don't waste
                # remaining calls in this batch if API is already broken
                if any(p in result_content for p in self._ENV_ERROR_PATTERNS):
                    break

            # Check for environment-side errors immediately after tool execution.
            # If API quota is exhausted, stop the evaluation now instead of
            # letting the model continue with broken tool data for more steps.
            env_error = self._detect_env_error(ep.tool_trace)
            if env_error:
                ep.done = True
                ep.final_score = 0.0
                ep.score_breakdown = {"error": env_error}
                return OpenEnvResponse(
                    observation="",
                    reward=0.0,
                    done=True,
                    episode_id=episode_id,
                    info={
                        "score": 0.0,
                        "score_breakdown": ep.score_breakdown,
                    },
                )

            # Standard OpenAI tool role: one message per tool result,
            # keyed by tool_call_id so the model can match request → response
            for r in tool_results:
                ep.conversation.append({
                    "role": "tool",
                    "tool_call_id": r["call_id"],
                    "content": r["result"],
                })

            observation = self._format_tool_results(tool_results)
            # Average step reward for this batch of tool calls
            avg_step_reward = sum(ep.step_rewards[-len(tool_calls):]) / len(tool_calls) if tool_calls else 0.0
            return OpenEnvResponse(
                observation=observation,
                reward=avg_step_reward,
                done=False,
                episode_id=episode_id,
                info={
                    "step": ep.current_step,
                    "tool_calls": len(tool_calls),
                    "step_reward": avg_step_reward,
                    "step_rewards_so_far": ep.step_rewards[:],
                },
            )
        else:
            # No tool calls or max steps reached, perform scoring
            ep.done = True

            # Check for environment-side errors (API rate limits, timeouts)
            # that are NOT the model's fault — invalidate the evaluation
            env_error = self._detect_env_error(ep.tool_trace)
            if env_error:
                ep.final_score = 0.0
                ep.score_breakdown = {"error": env_error}
                return OpenEnvResponse(
                    observation="完成",
                    reward=0.0,
                    done=True,
                    episode_id=episode_id,
                    info={
                        "score": 0.0,
                        "score_breakdown": ep.score_breakdown,
                    },
                )

            # scoring_input is the model's final answer.
            # evaluate() ensures this is a complete answer via the
            # dedicated final-answer phase, so no fallback needed.
            scoring_input = action or ""
            logger.debug("scoring_input: len=%d", len(scoring_input))

            try:
                score_result = await self._scorer.score(
                    scoring_input, ep.problem, ep.tool_trace
                )
                # QQR_DEBUG=1 → full breakdown for internal eval/debug
                # Default → safe (obfuscated) breakdown for external/RL consumption
                if os.getenv("QQR_DEBUG", "0") == "1":
                    ep.score_breakdown = score_result.to_dict()
                else:
                    ep.score_breakdown = score_result.to_safe_dict()
                ep._score_breakdown_full = score_result.to_dict()

                if score_result.llm_validation_error:
                    # No working judge → score is invalid. Force 0 + error so
                    # callers filter on error; code breakdown stays available
                    # for diagnostics.
                    ep.final_score = 0.0
                    ep.score_breakdown["error"] = (
                        f"LLM validator unavailable: {score_result.llm_validation_error}"
                    )
                else:
                    ep.final_score = score_result.total

                logger.debug("Scoring completed: total=%s, breakdown=%s", ep.final_score, ep.score_breakdown)
            except Exception as e:
                logger.error("Scoring failed: %s", e, exc_info=True)
                ep.final_score = 0.0
                ep.score_breakdown = {"error": str(e)}

            return OpenEnvResponse(
                observation="完成",
                reward=ep.final_score / 100.0,
                done=True,
                episode_id=episode_id,
                info={
                    "score": ep.final_score,
                    "score_breakdown": ep.score_breakdown
                },
            )

    async def evaluate(
        self,
        model: str = "moonshotai/Kimi-K2.5-TEE",
        base_url: str = "https://llm.chutes.ai/v1",
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        timeout: int = 300,
        temperature: float = 0.7,    # accepted for af-eval compat; affentctl has no --temperature flag yet
        api_key: Optional[str] = None,
        collect_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Drive a real agent loop via affent (Go binary at /usr/local/bin/affentctl).

        affent provides streaming, reasoning_content handling, transient-error retry,
        stream watchdog, and context compaction — none of which the old in-Python
        loop had. We spawn `affentctl run` per evaluation, read back the session
        JSONL + trace JSONL, populate the episode's conversation/tool_trace, then
        call step(tool_calls=None) to trigger scoring through the existing path.
        """
        del temperature  # not piped through to affent yet
        start_time = datetime.now()
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        task_id = task_id if task_id is not None else (seed & 0x7FFFFFFF)
        api_key = api_key or os.getenv("CHUTES_API_KEY")

        # Judge always uses CHUTES_API_KEY from env (with DashScope fallback inside
        # the validator). Do NOT reuse the agent's api_key here — when callers point
        # the agent at DashScope, that key won't authenticate against Chutes and
        # the whole judge chain collapses to "all models failed".
        if os.getenv("CHUTES_API_KEY") and not self._llm_evaluator:
            async with self._init_lock:
                if not self._llm_evaluator:
                    self._llm_evaluator = get_llm_evaluator()  # picks up CHUTES_API_KEY itself
                    if self._llm_evaluator:
                        self._scorer = TravelScorer(llm_validator=self._llm_evaluator)

        episode_id = None
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        async with self._eval_semaphore:
            reset_resp = await self.reset(task_id=task_id, seed=seed)
            episode_id = reset_resp.episode_id
            ep = self._episodes[episode_id]
            user_prompt = reset_resp.observation

            try:
                affent_result = await run_affent_agent(
                    user_prompt=user_prompt,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    epoch_salt=_EPOCH_SALT,
                    timeout=timeout,
                    max_turns=MAX_TOOL_STEPS + 2,
                )

                total_usage = affent_result["usage"]
                final_answer = affent_result["final_answer"]

                # affent's session JSONL is the canonical conversation: it includes
                # our SYSTEM_PROMPT (passed via --system-prompt), the user prompt
                # we sent on stdin, and every assistant/tool exchange.
                ep.conversation = affent_result["conversation"] or ep.conversation
                ep.tool_trace = affent_result["tool_trace"]

                if not final_answer.strip():
                    logger.warning(
                        "affent produced no final answer (exit=%s); stderr_tail=%s",
                        affent_result["exit_code"], affent_result["stderr_tail"][-512:],
                    )
                    return {
                        "task_name": "qqr",
                        "score": 0.0,
                        "success": False,
                        "time_taken": (datetime.now() - start_time).total_seconds(),
                        "extra": {
                            "error": "Agent produced no final answer",
                            "seed": seed,
                            "task_id": task_id,
                            "usage": total_usage,
                            "agent": "affent",
                            "tool_trace": ep.tool_trace,
                        },
                    }

                # Trigger scoring via the existing step() path (tool_calls=None branch
                # handles env_error detection + scorer.score() + ep.final_score).
                await self.step(
                    action=final_answer,
                    episode_id=episode_id,
                    tool_calls=None,
                )

                final_ep = self._episodes.get(episode_id)
                if not final_ep:
                    return {
                        "task_name": "qqr",
                        "score": 0.0,
                        "success": False,
                        "time_taken": (datetime.now() - start_time).total_seconds(),
                        "extra": {"error": "episode not found", "seed": seed, "task_id": task_id},
                    }

                score = final_ep.final_score
                conv = final_ep.conversation

                result = {
                    "task_name": "qqr",
                    "score": score / 100.0,
                    "success": score >= 60,
                    "time_taken": (datetime.now() - start_time).total_seconds(),
                    "extra": {
                        "conversation": conv,
                        "conversation_total_messages": len(conv),
                        "tool_trace": final_ep.tool_trace,
                        "seed": seed,
                        "task_id": task_id,
                        "usage": total_usage,
                        "score_raw": score,
                        "score_breakdown": final_ep.score_breakdown,
                        "problem": final_ep.problem.to_dict(),
                        "step_rewards": final_ep.step_rewards,
                        "avg_step_reward": (
                            sum(final_ep.step_rewards) / len(final_ep.step_rewards)
                            if final_ep.step_rewards else 0.0
                        ),
                        "agent": "affent",
                        "affent_exit_code": affent_result["exit_code"],
                    },
                }
                llm_error = final_ep.score_breakdown.get("error")
                if llm_error:
                    result["extra"]["error"] = llm_error

                if collect_logprobs and conv:
                    try:
                        from affinetes.core.logprobs_utils import collect_full_logprobs
                        full_logprobs = await collect_full_logprobs(
                            conversation=conv,
                            model=model,
                            base_url=base_url,
                            api_key=api_key,
                        )
                        result["extra"]["full_logprobs"] = full_logprobs
                    except Exception as e:
                        result["extra"]["full_logprobs"] = None
                        result["extra"]["logprobs_error"] = str(e)

                return result

            except FileNotFoundError as e:
                logger.error("affent binary missing: %s", e)
                return {
                    "task_name": "qqr",
                    "score": 0.0,
                    "success": False,
                    "time_taken": (datetime.now() - start_time).total_seconds(),
                    "extra": {
                        "error": f"affent binary not available: {e}",
                        "seed": seed,
                        "task_id": task_id,
                    },
                }
            except RuntimeError as e:
                logger.warning("affent run failed: %s", e)
                return {
                    "task_name": "qqr",
                    "score": 0.0,
                    "success": False,
                    "time_taken": (datetime.now() - start_time).total_seconds(),
                    "extra": {
                        "error": f"Agent run failed: {e}",
                        "seed": seed,
                        "task_id": task_id,
                        "usage": total_usage,
                    },
                }

            finally:
                if episode_id is not None:
                    self._episodes.pop(episode_id, None)
                self._eval_count += 1
                if self._eval_count % self._gc_interval == 0:
                    gc.collect()

    @staticmethod
    def _truncate_tool_result(content: str, max_chars: int = 2000) -> str:
        """Truncate tool result at JSON-structure boundaries.

        If the content is a JSON array, keeps complete items that fit.
        If it's a JSON object, keeps complete top-level keys.
        Falls back to raw truncation with a marker if not JSON.
        """
        if len(content) <= max_chars:
            return content

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON — truncate at last newline before limit
            cut = content[:max_chars].rfind('\n')
            if cut < max_chars // 2:
                cut = max_chars
            return content[:cut] + f"\n... (truncated, {len(content)} chars total)"

        # MCP TextContent: {"type":"text","text":"...","annotations":null,"meta":null}
        # Extract the "text" field's actual content, then truncate as plain text
        # (same logic as scorer._get_result_string for consistency)
        if (isinstance(data, dict)
                and data.get("type") == "text"
                and isinstance(data.get("text"), str)):
            inner_text = data["text"]
            if len(inner_text) <= max_chars:
                return inner_text
            cut = inner_text[:max_chars].rfind('\n')
            if cut < max_chars // 2:
                cut = max_chars
            return inner_text[:cut] + f"\n... (truncated, {len(inner_text)} chars total)"

        if isinstance(data, list):
            # Keep complete items that fit within budget
            kept = []
            current_len = 2  # for "[]"
            for item in data:
                item_json = json.dumps(item, ensure_ascii=False)
                if current_len + len(item_json) + 2 > max_chars:  # +2 for ", "
                    break
                kept.append(item)
                current_len += len(item_json) + 2
            if len(kept) < len(data):
                result = json.dumps(kept, ensure_ascii=False, indent=None)
                return result + f"\n... ({len(kept)}/{len(data)} items shown)"
            return json.dumps(kept, ensure_ascii=False, indent=None)

        elif isinstance(data, dict):
            # Try to keep as much of the dict as possible
            result_json = json.dumps(data, ensure_ascii=False, indent=None)
            if len(result_json) <= max_chars:
                return result_json
            # Remove keys from the end until it fits
            keys = list(data.keys())
            kept = {}
            current_len = 2  # for "{}"
            for k in keys:
                entry = json.dumps({k: data[k]}, ensure_ascii=False)
                if current_len + len(entry) > max_chars:
                    break
                kept[k] = data[k]
                current_len += len(entry)
            result = json.dumps(kept, ensure_ascii=False, indent=None)
            return result + f"\n... ({len(kept)}/{len(keys)} fields shown)"

        # Scalar or other type
        return content[:max_chars]

    # Environment-side errors that invalidate the evaluation.
    # These are NOT the model's fault — API quota exhaustion, infra failures, etc.
    _ENV_ERROR_PATTERNS = [
        "USER_DAILY_QUERY_OVER_LIMIT",
        "DAILY_QUERY_OVER_LIMIT",
        "QPS_OVER_LIMIT",
        "USERBAND_BINDbindedkeysoverrun",
        "CUOTA_PLAN_RUN_OUT",
        "INVALID_USER_KEY",
        "INVALID_USER_SCODE",
        "tool_call_timeout",
    ]

    def _detect_env_error(self, tool_trace: List[Dict]) -> Optional[str]:
        """Detect environment-side errors in tool trace.

        Returns error description if any tool call hit an API-side error
        (rate limit, quota exhaustion, infra timeout), or None if clean.
        Only flags errors that are the environment's fault, not the model's
        (e.g., INVALID_PARAMS is the model's fault, OVER_LIMIT is ours).
        """
        env_errors = []
        for t in tool_trace:
            result = t.get("result", {})
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            for pattern in self._ENV_ERROR_PATTERNS:
                if pattern in text:
                    env_errors.append(f"{t['name']}: {pattern}")
                    break

        if not env_errors:
            return None

        return f"Environment error (not model fault): {len(env_errors)} tool call(s) failed due to API limit — {env_errors[0]}"

    def _format_tool_results(self, results: List[Dict]) -> str:
        """Format tool return results."""
        return "\n\n".join(
            f"[{r['tool']}] 结果:\n{r['result']}"
            for r in results
        )

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Return container memory stats from /proc/self/status."""
        rss_kb = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except OSError:
            pass
        stats = {
            "rss_mb": round(rss_kb / 1024, 1),
            "episodes_active": len(self._episodes),
            "eval_count": self._eval_count,
        }
        # Include cache stats if MCP servers are initialized
        if self._mcp_state and self._mcp_state._mcp_servers:
            for server in self._mcp_state._mcp_servers:
                if hasattr(server, 'get_cache_stats'):
                    stats[f"cache_{server.name}"] = server.get_cache_stats()
        return stats

    async def cleanup(self):
        """Clean up resources — close MCP servers and release memory."""
        # Clear any lingering episodes
        self._episodes.clear()
        # Shut down MCP servers
        if self._mcp_state:
            try:
                await self._mcp_state.cleanup()
            except Exception as e:
                logger.warning("MCPState cleanup error: %s", e)
            self._mcp_initialized = False
        gc.collect()

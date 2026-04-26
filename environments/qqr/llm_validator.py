"""
LLM Validator - Modular LLM scoring framework with abstract scorer groups.

Architecture:
  LLMScorerGroup (ABC) — base class for scorer groups with model fallback + retry
    └─ UnifiedScorer — 5 dimensions in single call

  LLMEvaluator — orchestrates execution + cross-validation
"""

import asyncio
import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from openai import AsyncOpenAI

from problem_generator import TravelProblem


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LLMEvaluationResult:
    """Combined result from all scorer groups."""

    # Quality dimensions (0-10 scale)
    practicality: float = 0.0
    analysis_depth: float = 0.0
    logic: float = 0.0
    user_experience: float = 0.0

    # Grounding dimension (0-10 scale)
    factual_grounding: float = 0.0

    # Per-group success flags
    quality_success: bool = False
    grounding_success: bool = False

    # Metadata
    quality_model: Optional[str] = None
    grounding_model: Optional[str] = None
    quality_reasons: Dict[str, str] = field(default_factory=dict)
    grounding_reason: str = ""
    error: str = ""

    @property
    def success(self) -> bool:
        return self.quality_success or self.grounding_success

    def to_dict(self) -> dict:
        return {
            "practicality": self.practicality,
            "analysis_depth": self.analysis_depth,
            "logic": self.logic,
            "user_experience": self.user_experience,
            "factual_grounding": self.factual_grounding,
            "quality_success": self.quality_success,
            "grounding_success": self.grounding_success,
            "quality_model": self.quality_model,
            "grounding_model": self.grounding_model,
            "quality_reasons": self.quality_reasons,
            "grounding_reason": self.grounding_reason,
            "error": self.error,
        }


# For backward compat — scorer.py references this
@dataclass
class LLMValidationResult:
    """Legacy result format for backward compatibility."""
    tool_info_used: bool = False
    tool_usage_reason: str = ""
    practicality: float = 0.0
    analysis_depth: float = 0.0
    logic: float = 0.0
    user_experience: float = 0.0
    reasons: Dict[str, str] = field(default_factory=dict)
    raw_response: str = ""
    success: bool = False
    error: str = ""

    @property
    def total(self) -> float:
        return self.practicality + self.analysis_depth + self.logic + self.user_experience

    def to_dict(self) -> dict:
        return {
            "tool_info_used": self.tool_info_used,
            "tool_usage_reason": self.tool_usage_reason,
            "practicality": self.practicality,
            "analysis_depth": self.analysis_depth,
            "logic": self.logic,
            "user_experience": self.user_experience,
            "total": self.total,
            "reasons": self.reasons,
            "success": self.success,
            "error": self.error,
        }


# ============================================================================
# Prompts
# ============================================================================

# Unified evaluation prompt: quality + factual grounding in a single call
# NOTE: LLM never sees raw model output — only code-extracted structured summary.
# This eliminates the prompt injection attack surface entirely.
UNIFIED_EVALUATION_PROMPT = '''你是旅游规划质量与事实核查评估专家。请根据以下由代码提取的结构化摘要评估旅行规划的质量和事实准确性。

=== 评分校准 ===
请注意：5分代表"中等水平"，不是"差"。大多数普通规划应得5-6分。
7-8分应保留给明显高于平均水平的维度。9-10分极为罕见。
如果你觉得五个维度平均分超过7分，请重新检查是否对标准要求够严格。
{calibration_anchor}

=== 重要评分原则 ===
请特别注意区分两种模型行为：
1.「数据搬运」：将工具返回的信息原样列出，仅做简单排列，缺乏分析和整合
2.「深度分析」：基于工具数据进行推理，解释为什么推荐，分析利弊权衡，给出个性化建议

数据搬运行为应在各维度获低分，尤其是分析深度维度。
如果推理分析基于编造数据（工具未返回的信息），则分析深度和逻辑连贯性也应相应扣分。

=== 用户需求 ===
出发城市: {origin_city}
目的地: {destination_city}
旅行日期: {travel_date}
旅行天数: {num_days}天
预算: {budget}元
偏好: {preference}
兴趣: {interests}
约束: {constraints}

=== 工具返回的可验证事实 ===
{facts_summary}

=== 模型输出结构化摘要（由代码提取，非模型原文）===
以下信息均由评测代码从模型输出中自动提取，不包含模型原始文本。
{structured_summary}

=== 评估要求 ===

请根据以上结构化摘要评估以下五个维度。

{dimension_blocks}

=== 输出格式 ===

请严格输出以下JSON格式（不要输出其他内容）：

```json
{{
  "practicality": {{"score": <0-10>, "reason": "<说明>"}},
  "analysis_depth": {{"score": <0-10>, "reason": "<说明>"}},
  "logic": {{"score": <0-10>, "reason": "<说明>"}},
  "user_experience": {{"score": <0-10>, "reason": "<说明>"}},
  "factual_grounding": {{"score": <0-10>, "reason": "<说明>"}}
}}
```'''


# Dimension block templates with parameterized penalties
_DIM_BLOCKS = {
    "practicality": '''【{dim_label}: 规划可行性 practicality】(0-10分)
评分方法：从8分起评，按以下问题扣分（最低0分）：
- 未说明景点间交通方式或耗时 → -{penalty_transport}分
- 未给出每日具体时间段（如"上午9点""下午2点"） → -{penalty_time}分
- 出现时间冲突（同时段安排两个活动）→ -{penalty_conflict}分
- 跨城行程未安排城际交通 → -{penalty_cross_city}分

加分项（最高10分）：
- 每段交通都有具体方式+耗时 → +1分
- 时间安排精确到小时且无冲突 → +1分''',
    "analysis_depth": '''【{dim_label}: 分析深度 analysis_depth】(0-10分)
- 9-10分: 每个推荐都有具体数据支撑的理由，有明确利弊对比，基于约束做了取舍分析
- 7-8分: 多数推荐有理由，部分分析较浅
- 5-6分: 约一半推荐有分析，另一半是简单罗列
- 3-4分: 主要是数据罗列，分析浮于表面
- 0-2分: 纯数据搬运，零分析

重要判断标准——区分"数据复述"与"真正分析"：
- "推荐航班X因为它Y点出发"只是复述工具数据，不是分析。即使加了"因为/考虑到/综合"等连接词，本质仍是数据搬运。
- 真正的分析应解读数据的含义，例如："08:30出发意味着不需要起太早，到达后有充分游览时间"。
- 在工具仅返回少量方案时声称"经过仔细对比和权衡"是空洞表述（结构化摘要中会标注可对比方案数量）。
- 诚实指出工具返回有限（如"仅返回1个景点"）并给出实际建议，比虚假声称多方对比更有分析价值。
- 如果分析基于编造数据（工具未返回的航班、价格、距离等），应视为无效分析，相应扣分。

加分项（在基础评分上额外加分，最高10分）：
- 主动指出工具返回数据的局限性并提出补救建议（结构化摘要中"数据局限性认知"标记为✓）→ +2分
- 用具体数据计算佐证观点（如加总预算、计算时间窗口）→ +1分''',
    "logic": '''【{dim_label}: 逻辑连贯性 logic】(0-10分)
评分方法：从8分起评，按以下问题扣分（最低0分）：
- 相邻景点不在同一区域（不必要的跨区移动）→ -{penalty_cross_district}分
- 出现不必要的折返 → -{penalty_backtracking}分
- 景点安排无说明顺序理由 → -{penalty_no_reason}分
- 地理方位完全不合理 → -{penalty_geography}分
- 基于编造的距离/时间进行路线规划 → 逻辑推导无效，额外扣2分
- 基于工具未返回的POI进行地理分组或区域规划 → -{penalty_ungrounded_poi_logic}分（参见结构化摘要中"无工具依据的POI"列表，如果路线安排涉及这些POI，其地理逻辑不应获得正分）

加分项（最高10分）：
- 明确说明按地理位置/区域分组安排（仅限工具返回的POI）→ +1分
- 路线形成合理的单向或环形 → +1分''',
    "user_experience": '''【{dim_label}: 用户体验 user_experience】(0-10分)
- 9-10分: 所有约束和偏好明确回应，预算分配合理且有说明，矛盾约束有权衡
- 7-8分: 大部分需求已回应，1-2个约束未体现
- 5-6分: 回应了核心需求，但多个约束被忽略
- 3-4分: 仅部分考虑，通用模板感明显
- 0-2分: 完全忽视用户需求''',
    "factual_grounding": '''【{dim_label}: 事实准确性 factual_grounding】(0-10分)
从10分起评，逐项扣分（最低0分）：
- 输出中出现工具未返回的航班号/车次 → -{penalty_ungrounded_transport}分/个
- 未调用direction工具却声称两地之间的具体行车/步行耗时或公里距离 → -{penalty_ungrounded_time_distance}分
  （注意：航班/火车的出发到达时间来自交通查询工具，不属于此项扣分范围）
- 出现工具完全未返回的价格 → -{penalty_fabricated_price}分/处
  （注意：工具返回价格的简单加减合计不算编造，如850+150=1000属于可推导数据）
- 推荐的POI完全不在工具返回列表中 → -{penalty_ungrounded_poi}分/2个
- 输出过于简短或几乎未引用任何工具返回的具体数据（如景点名、航班号、价格、天气等） → 基础分不超过3分
加分项：所有事实均可在工具数据中找到出处 → 维持满分''',
}

NUM_FIXED_JUDGES = 3

# ============================================================================
# Abstract Base Class
# ============================================================================

class LLMScorerGroup(ABC):
    """LLM scorer group base class. Each group evaluates a set of related dimensions."""

    def __init__(
        self,
        models: List[str],
        client: AsyncOpenAI,
        timeout: int = 60,
        max_rounds: int = 10,
        # Deprecated: kept for backward compat, no longer used
        retries_per_model: int = 1,
    ):
        self.models = models
        self.client = client
        self.timeout = timeout
        self.max_rounds = max_rounds

    @property
    @abstractmethod
    def group_name(self) -> str:
        """Group identifier, e.g. 'quality', 'grounding'"""

    @property
    @abstractmethod
    def dimension_names(self) -> List[str]:
        """List of dimension names this group evaluates."""

    @abstractmethod
    def build_prompt(self, context: Dict[str, Any]) -> str:
        """Build evaluation prompt with targeted context."""

    @abstractmethod
    def parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response. Returns {dim: {"score": float, "reason": str}} or None."""

    async def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluation: round-robin across models for max_rounds.

        Strategy: cycle through all models repeatedly. Chutes models have
        intermittent availability, so spreading attempts across rounds
        maximises the chance that at least one model is healthy.
        Total attempts = len(models) * max_rounds.
        """
        prompt = self.build_prompt(context)
        total_attempts = 0

        parsed_results = []

        for _ in range(NUM_FIXED_JUDGES):
            for round_idx in range(self.max_rounds):
                judged = False

                for model in self.models:
                    total_attempts += 1
                    try:
                        response = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0,
                                max_tokens=2000,
                            ),
                            timeout=self.timeout,
                        )
                        content = response.choices[0].message.content
                        parsed = self.parse_response(content)
                        if parsed is not None:
                            parsed["_model"] = model
                            parsed["_success"] = True
                            parsed_results.append(parsed)
                            judged = True
                            break
                        # Parse failed — treat as transient, try next model
                        print(f"[{self.group_name}] {model} parse failed (round {round_idx+1})")
                    except asyncio.TimeoutError:
                        print(f"[{self.group_name}] {model} timeout (round {round_idx+1})")
                    except Exception as e:
                        print(f"[{self.group_name}] {model} error: {e} (round {round_idx+1})")

                    # Brief backoff between attempts; longer between rounds
                    await asyncio.sleep(1 if round_idx == 0 else 2)
                
                if judged:
                    break  # Move to next fixed judge after a successful evaluation
            
                print(f"[{self.group_name}] round {round_idx+1}/{self.max_rounds} exhausted")

            print(f"[{self.group_name}] - {len(parsed_results)}/{NUM_FIXED_JUDGES} judges succeeded")
        
        final_result = {}
        final_result["_success"] = len(parsed_results) > 0
        final_result["_model"] = parsed_results[0].get("_model") if parsed_results else None

        for dim in self.dimension_names:
            avg_score = 0.0
            reason = parsed_results[0].get(dim, {}).get("reason", "") if parsed_results else ""

            for parsed_result in parsed_results:
                if dim in parsed_result:
                    avg_score += parsed_result[dim]["score"]
            
            final_result[dim] = {"score": avg_score / len(parsed_results), "reason": reason}

        return final_result if len(parsed_results) > 0 else {dim: {"score": 0.0, "reason": "all models failed"}
                for dim in self.dimension_names} | {"_success": False, "_model": None}


# ============================================================================
# UnifiedScorer — 5 dimensions in a single call
# ============================================================================

class UnifiedScorer(LLMScorerGroup):
    """Evaluates all 5 dimensions in a single LLM call:
    practicality, analysis_depth, logic, user_experience, factual_grounding."""

    @property
    def group_name(self) -> str:
        return "unified"

    @property
    def dimension_names(self) -> List[str]:
        return ["practicality", "analysis_depth", "logic", "user_experience", "factual_grounding"]

    def build_prompt(self, context: Dict[str, Any]) -> str:
        """Build unified evaluation prompt with all 5 dimensions.

        Key design: LLM never sees raw model output. Instead we pass a
        code-extracted structured summary, eliminating injection surface.
        """
        from config import LLM_RUBRIC_PENALTY_RANGES
        import random as _rand

        problem = context["problem"]
        model_output = context.get("model_output", "")
        tool_facts = context.get("tool_facts")
        called_tools = context.get("called_tools", set())

        facts_summary = _format_facts_summary(tool_facts, context.get("output_facts"), called_tools)
        # Build structured summary instead of passing raw output
        structured_summary = _build_structured_summary(
            model_output, tool_facts, called_tools, problem
        )

        # Parameterized rubric: randomize penalties per task+epoch
        epoch_salt = os.getenv("TRANSPORT_SALT", "default")
        rng = _rand.Random(f"{problem.task_id}_{epoch_salt}_rubric")
        penalties = {k: rng.randint(*v) for k, v in LLM_RUBRIC_PENALTY_RANGES.items()}

        # Randomize dimension order (all 5 dimensions)
        dim_order = list(self.dimension_names)
        rng.shuffle(dim_order)

        dim_labels = {d: f"维度{i+1}" for i, d in enumerate(dim_order)}
        dim_blocks_parts = []
        for dim in dim_order:
            block = _DIM_BLOCKS[dim].format(
                dim_label=dim_labels[dim],
                penalty_transport=penalties.get("no_transport_mode", 3),
                penalty_time=penalties.get("no_time_slots", 2),
                penalty_conflict=penalties.get("time_conflict", 4),
                penalty_cross_city=penalties.get("cross_city_no_transport", 4),
                penalty_cross_district=penalties.get("cross_district", 3),
                penalty_backtracking=penalties.get("backtracking", 2),
                penalty_no_reason=penalties.get("no_order_reason", 2),
                penalty_geography=penalties.get("bad_geography", 4),
                penalty_ungrounded_transport=penalties.get("ungrounded_transport", 4),
                penalty_ungrounded_time_distance=penalties.get("ungrounded_time_distance", 3),
                penalty_fabricated_price=penalties.get("fabricated_price", 3),
                penalty_ungrounded_poi=penalties.get("ungrounded_poi", 2),
                penalty_ungrounded_poi_logic=penalties.get("ungrounded_poi_logic", 3),
            )
            dim_blocks_parts.append(block)

        anchors = [
            "注意：此类旅行规划通常得4-6分，高于7分需有充分理由。",
            "参考：一般水平的规划在各维度约5分左右。",
            "校准提示：请确保评分分布合理，避免集中在高分段。",
        ]
        calibration_anchor = rng.choice(anchors)

        return UNIFIED_EVALUATION_PROMPT.format(
            origin_city=problem.origin_city or "N/A",
            destination_city=problem.destination_city,
            travel_date=problem.travel_date,
            num_days=problem.num_days,
            budget=problem.budget or "不限",
            preference=problem.preference or "无特殊偏好",
            interests=", ".join(problem.interests) if problem.interests else "无特定兴趣",
            constraints=", ".join(problem.constraints) if problem.constraints else "无特殊约束",
            facts_summary=facts_summary,
            structured_summary=structured_summary,
            dimension_blocks="\n\n".join(dim_blocks_parts),
            calibration_anchor=calibration_anchor,
        )

    def parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse unified scorer response into 5 dimension scores."""
        try:
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
            result = {}
            for dim in self.dimension_names:
                score, reason = _extract_dimension_score(data.get(dim, 0))
                result[dim] = {"score": min(10, max(0, score)), "reason": reason}
            return result
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None


# ============================================================================
# LLMEvaluator — orchestrates execution + cross-validation
# ============================================================================

class LLMEvaluator:
    """Orchestrates unified scorer execution with cross-validation."""

    def __init__(
        self,
        client: AsyncOpenAI,
        models: List[str],
        # Backward compat kwargs (ignored, models param used for all)
        quality_models: Optional[List[str]] = None,
        grounding_models: Optional[List[str]] = None,
    ):
        # If called with old-style kwargs, use quality_models as fallback
        effective_models = models or quality_models or []
        self.unified = UnifiedScorer(models=effective_models, client=client)

    async def evaluate(self, context: Dict[str, Any]) -> LLMEvaluationResult:
        """Execute unified scorer, merge results."""
        result = LLMEvaluationResult()

        if not context.get("tool_trace"):
            result.error = "No tools called, cannot evaluate"
            return result

        # Single unified call for all 5 dimensions
        unified_result = await self.unified.evaluate(context)

        if isinstance(unified_result, Exception):
            result.error = f"Unified scorer exception: {unified_result}"
            return result

        if unified_result.get("_success", False):
            model_used = unified_result.get("_model")
            # Set both success flags atomically
            result.quality_success = True
            result.grounding_success = True
            result.quality_model = model_used
            result.grounding_model = model_used

            # Quality dimensions
            for dim in ["practicality", "analysis_depth", "logic", "user_experience"]:
                dim_data = unified_result.get(dim, {})
                setattr(result, dim, dim_data.get("score", 0.0))
                result.quality_reasons[dim] = dim_data.get("reason", "")

            # Grounding dimension
            fg_data = unified_result.get("factual_grounding", {})
            result.factual_grounding = fg_data.get("score", 0.0)
            result.grounding_reason = fg_data.get("reason", "")

            # Grounding-quality coupling: if fg is very low, the reasoning
            # is built on fabricated data → cap logic and analysis_depth.
            # This prevents attackers from writing "convincing" reasoning
            # based on made-up facts to score high on quality dimensions.
            result = self._apply_grounding_coupling(result)

            # Cross-validation for high scores (all 5 dimensions)
            result = await self._cross_validate(context, result)
        else:
            result.error = "Unified scorer: all models failed"

        return result

    @staticmethod
    def _apply_grounding_coupling(result: LLMEvaluationResult) -> LLMEvaluationResult:
        """Cap logic and analysis_depth when factual_grounding is very low.

        Rationale: if the LLM itself judges that facts are fabricated (fg<=3),
        then any "good reasoning" built on those facts is invalid.
        """
        fg = result.factual_grounding
        if fg <= 3.0:
            # Hard cap: logic and analysis_depth cannot exceed fg + 2
            cap = fg + 2.0
            if result.logic > cap:
                result.quality_reasons["logic"] = (
                    result.quality_reasons.get("logic", "") +
                    f" [grounding-coupled: capped from {result.logic:.1f} to {cap:.1f}]"
                )
                result.logic = cap
            if result.analysis_depth > cap:
                result.quality_reasons["analysis_depth"] = (
                    result.quality_reasons.get("analysis_depth", "") +
                    f" [grounding-coupled: capped from {result.analysis_depth:.1f} to {cap:.1f}]"
                )
                result.analysis_depth = cap
        return result

    async def _cross_validate(
        self, context: Dict[str, Any], primary: LLMEvaluationResult
    ) -> LLMEvaluationResult:
        """Cross-validate high scores with a different model (all 5 dimensions)."""
        total = (
            primary.practicality + primary.analysis_depth +
            primary.logic + primary.user_experience + primary.factual_grounding
        )
        if total <= 36:  # <=72% of 50 max → skip
            return primary

        # Use a different model for cross-validation
        cross_models = [m for m in self.unified.models if m != primary.quality_model]
        if not cross_models:
            return primary

        cross_scorer = UnifiedScorer(
            models=cross_models,
            client=self.unified.client,
            max_rounds=1,
            timeout=self.unified.timeout,
        )
        cross_result = await cross_scorer.evaluate(context)
        if not cross_result.get("_success", False):
            return primary

        # Take minimum of each dimension (conservative) — all 5 dimensions
        for dim in self.unified.dimension_names:
            primary_val = getattr(primary, dim)
            cross_val = cross_result.get(dim, {}).get("score", primary_val)
            setattr(primary, dim, min(primary_val, cross_val))

        return primary


# ============================================================================
# Backward compat aliases
# ============================================================================

class QualityScorer(UnifiedScorer):
    """Backward compat alias for UnifiedScorer."""
    pass


class GroundingScorer(UnifiedScorer):
    """Backward compat alias for UnifiedScorer."""
    pass


# ============================================================================
# Shared Utility Functions
# ============================================================================

def _sanitize_output_for_validation(raw_output: str) -> str:
    """Extract factual content, filtering injection attempts."""
    text = raw_output[:15000]

    # Layer 1: Remove control characters and invisible Unicode
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]', '', text)

    # Layer 2: Expanded injection pattern blacklist
    injection_patterns = [
        r'(?i)(?:请|please)?\s*(?:忽略|ignore)\s*(?:以上|above|previous)',
        r'(?i)(?:将|set)\s*(?:所有|all)\s*(?:分数|score)',
        r'(?i)(?:你是|you are)\s*(?:一个|a)\s*(?:评分|scoring)',
        r'(?i)(?:system|系统)\s*(?:prompt|提示)',
        r'(?i)(?:override|覆盖)\s*(?:instructions?|指令)',
        r'(?i)给\s*(?:满分|最高分|10分)',
        r'(?i)(?:评分|score)\s*(?:标准|rubric|criteria)',
        r'(?i)(?:json|JSON)\s*(?:格式|format)\s*(?:如下|below)',
        r'(?i)\{["\']?(?:practicality|logic|analysis_depth|user_experience)',
        r'(?i)(?:reason|理由)\s*[:：]\s*["\']',
        r'(?i)(?:assistant|AI|人工智能)\s*(?:注意|note|notice)',
        r'(?i)(?:以下|following)\s*(?:是|is)\s*(?:评估|evaluation)',
        r'(?i)(?:根据|according)\s*(?:评分|scoring)\s*(?:标准|standard)',
        r'(?i)(?:维度|dimension)\s*\d',
        r'(?i)(?:扣分|deduct|penalty)',
        # Soft injection patterns: evaluation hints / self-scoring
        r'(?i)(?:评估|评分)\s*(?:专家|系统|模型)\s*(?:参考|注意|建议)',
        r'(?i)(?:请|建议)\s*(?:评估|评分)\s*(?:专家|者)',
        r'(?i)(?:达到|获得|给予)\s*(?:优秀|满分|高分|最高)',
        r'(?i)(?:审核|认证|验证)\s*(?:通过|合格|优秀)',
        r'(?i)(?:专业|权威)\s*(?:旅行社|机构|认证)',
        r'(?i)(?:建议|应当|请)\s*(?:给予|给出)\s*(?:高|最高|满)',
        r'(?i)\b\d+\s*分\s*(?:以上|左右).*(?:维度|评分)',
        r'(?i)(?:自我|自评|self).{0,10}(?:评价|评估|assessment)',
        r'(?i)(?:factual_grounding|practicality|analysis_depth|logic|user_experience)\s*[=：:]\s*\d',
        r'(?i)\[SYSTEM\]',
        r'(?i)模式已切换',
    ]

    lines = text.split('\n')
    filtered = [l for l in lines if not any(re.search(p, l) for p in injection_patterns)]
    return '\n'.join(filtered)


def _extract_dimension_score(val) -> tuple:
    """Extract score from dimension data, handling both dict and bare number."""
    if isinstance(val, dict):
        return float(val.get("score", 0)), val.get("reason", "")
    elif isinstance(val, (int, float)):
        return float(val), ""
    return 0.0, ""


def _get_result_text(result) -> str:
    """Extract text from tool result, handling double-nested JSON."""
    if isinstance(result, dict):
        text = result.get("text", json.dumps(result, ensure_ascii=False))
        if isinstance(text, str) and text.startswith('{'):
            try:
                inner = json.loads(text)
                if isinstance(inner, dict) and "text" in inner:
                    return inner["text"]
            except (json.JSONDecodeError, TypeError):
                pass
        return text
    return str(result)


def _format_tool_trace(tool_trace: List[Dict]) -> str:
    """Format tool call records for quality evaluation prompt."""
    if not tool_trace:
        return "（无工具调用记录）"

    lines = []
    key_info = {"poi_names": [], "flights": [], "trains": []}

    for i, call in enumerate(tool_trace, 1):
        name = call.get("name", "unknown")
        args = call.get("arguments", {})
        result = call.get("result", {})
        text = _get_result_text(result)

        lines.append(f"【调用{i}】{name}")
        lines.append(f"  参数: {json.dumps(args, ensure_ascii=False)[:200]}")

        if name == "poi_search":
            lines.append(f"  返回: {text[:500]}...")
            poi_matches = re.findall(r'(?:名称|name)[：:]\s*([^\n,，]{2,40})', text)
            key_info["poi_names"].extend(poi_matches)
        elif name == "search_flights":
            lines.append(f"  返回: {text[:500]}...")
            flight_matches = re.findall(r'航班\s*([A-Z0-9]+)', text)
            key_info["flights"].extend(flight_matches)
        elif name == "search_train_tickets":
            lines.append(f"  返回: {text[:500]}...")
            train_matches = re.findall(r'车次\s*([GDCZTK]\d+)', text)
            key_info["trains"].extend(train_matches)
        elif name == "around_search":
            lines.append(f"  返回: {text[:500]}...")
            poi_matches = re.findall(r'(?:名称|name)[：:]\s*([^\n,，]{2,40})', text)
            key_info["poi_names"].extend(poi_matches)
        elif name == "direction":
            lines.append(f"  返回: {text[:300]}...")
        elif name == "weather":
            lines.append(f"  返回: {text[:200]}...")
        else:
            lines.append(f"  返回: {text[:200]}...")

        lines.append("")

    summary = []
    if key_info["poi_names"]:
        summary.append(f"★ 工具返回的POI名称: {key_info['poi_names'][:10]}")
    if key_info["flights"]:
        summary.append(f"★ 工具返回的航班号: {key_info['flights'][:10]}")
    if key_info["trains"]:
        summary.append(f"★ 工具返回的车次: {key_info['trains'][:10]}")

    if summary:
        lines.insert(0, "=== 关键信息汇总 ===\n" + "\n".join(summary) + "\n")

    return "\n".join(lines)


def _format_facts_summary(tool_facts, output_facts, called_tools: Set[str]) -> str:
    """Format structured facts summary for grounding evaluation."""
    if tool_facts is None:
        return "（无可验证事实）"

    sections = []
    if tool_facts.pois:
        sorted_pois = sorted(tool_facts.pois)[:15]
        sections.append(f"工具返回POI ({len(tool_facts.pois)}个): {', '.join(sorted_pois)}")
    if tool_facts.flights:
        sections.append(f"工具返回航班号: {', '.join(sorted(tool_facts.flights)[:10])}")
    if tool_facts.trains:
        sections.append(f"工具返回车次: {', '.join(sorted(tool_facts.trains)[:10])}")
    if tool_facts.transport_prices:
        ps = [f"{k}:{v}元" for k, v in sorted(tool_facts.transport_prices.items())[:8]]
        sections.append(f"工具返回交通价格: {', '.join(ps)}")
    # Include ALL unique price values (POI costs, ticket prices, etc.)
    if tool_facts.prices:
        unique_prices = sorted(set(int(v) for v in tool_facts.prices.values() if v > 0))
        if unique_prices:
            sections.append(f"工具返回所有价格值: {', '.join(f'{p}元' for p in unique_prices)}（注：这些价格的加减合计也属于可推导数据）")
    if tool_facts.times:
        sections.append(f"工具返回出发/到达时间: {', '.join(sorted(tool_facts.times)[:8])}")
    if tool_facts.weather:
        sections.append(f"工具返回天气: {', '.join(sorted(tool_facts.weather)[:6])}")
    if tool_facts.travel_durations:
        sections.append(f"工具返回行程时间: {', '.join(sorted(tool_facts.travel_durations)[:6])}")
    if tool_facts.distances:
        sections.append(f"工具返回距离: {', '.join(sorted(tool_facts.distances)[:6])}")

    # Critical annotations
    if "direction" not in called_tools:
        sections.append("⚠ 未调用direction工具（输出中的行程时间/距离均无数据来源）")
    if "search_flights" not in called_tools and "search_train_tickets" not in called_tools:
        sections.append("⚠ 未调用交通查询工具（输出中的航班/车次均无数据来源）")

    return "\n".join(sections) if sections else "（无可验证事实）"


# ============================================================================
# Structured Summary Builder — eliminates prompt injection surface
# ============================================================================

def _build_structured_summary(
    model_output: str,
    tool_facts,
    called_tools: Set[str],
    problem: 'TravelProblem',
) -> str:
    """Build a structured summary of model output using code extraction only.

    The LLM evaluator never sees raw model output. Instead, this function
    extracts structured facts, reasoning indicators, and coverage metrics
    from the output using regex/code, and formats them into a clean summary.
    This eliminates the prompt injection attack surface entirely.
    """
    from scorer import FactExtractor, ExtractedFacts

    text = model_output or ""
    text_lower = text.lower()
    extractor = FactExtractor()
    output_facts = extractor.extract_from_output(text)

    sections = []

    # ── 1. Basic statistics ──
    char_count = len(text)
    line_count = len(text.strip().split('\n')) if text.strip() else 0
    # Count days mentioned
    arabic_days = set(re.findall(r'第\s*(\d+)\s*天', text))
    chinese_days_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
    for match in re.finditer(r'第\s*([一二三四五六七八九十]+)\s*天', text):
        cn = match.group(1)
        if cn in chinese_days_map:
            arabic_days.add(str(chinese_days_map[cn]))
    day_n = set(re.findall(r'Day\s*(\d+)', text, re.IGNORECASE))
    days_mentioned = len(set(int(d) for d in arabic_days) | set(int(d) for d in day_n))

    sections.append(f"【基础统计】")
    sections.append(f"- 输出长度: {char_count} 字符, {line_count} 行")
    sections.append(f"- 天数规划: 提及{days_mentioned}天 (目标{problem.num_days}天)")

    # ── 2. Transport information with grounding check ──
    sections.append(f"\n【交通信息】")
    tool_flights = tool_facts.flights if tool_facts else set()
    tool_trains = tool_facts.trains if tool_facts else set()
    tool_transport_prices = tool_facts.transport_prices if tool_facts else {}

    if output_facts.flights:
        flight_items = []
        for f in sorted(output_facts.flights):
            grounded = "✓工具返回" if f in tool_flights else "✗未在工具数据中"
            flight_items.append(f"{f} ({grounded})")
        sections.append(f"- 提及航班: {', '.join(flight_items)}")
    else:
        sections.append(f"- 提及航班: 无")

    if output_facts.trains:
        train_items = []
        for t in sorted(output_facts.trains):
            grounded = "✓工具返回" if t in tool_trains else "✗未在工具数据中"
            train_items.append(f"{t} ({grounded})")
        sections.append(f"- 提及车次: {', '.join(train_items)}")
    else:
        sections.append(f"- 提及车次: 无")

    # Transport prices
    if output_facts.transport_prices:
        price_items = []
        for tid, price in sorted(output_facts.transport_prices.items()):
            tool_price = tool_transport_prices.get(tid)
            if tool_price is not None:
                diff_pct = abs(price - tool_price) / max(1, tool_price) * 100
                if diff_pct <= 20:
                    grounded = f"✓工具返回{tool_price}元"
                else:
                    grounded = f"✗工具返回{tool_price}元,偏差{diff_pct:.0f}%"
            else:
                grounded = "✗无工具来源"
            price_items.append(f"{tid}={price}元 ({grounded})")
        sections.append(f"- 交通价格: {', '.join(price_items[:8])}")

    # Time slots
    if output_facts.times:
        time_grounded = []
        tool_times = tool_facts.times if tool_facts else set()
        for t in sorted(output_facts.times):
            g = "✓" if t in tool_times else "○自行安排"
            time_grounded.append(f"{t}({g})")
        sections.append(f"- 时间安排: {', '.join(time_grounded[:10])}")

    # ── 3. POI / Attractions ──
    sections.append(f"\n【景点/POI信息】")
    tool_pois = tool_facts.pois if tool_facts else set()
    # Use fuzzy matching for POI names (substring match)
    if tool_pois:
        mentioned_pois = _extract_mentioned_pois(text, tool_pois)
        grounded_pois = [p for p in mentioned_pois if p["grounded"]]
        ungrounded_pois = [p for p in mentioned_pois if not p["grounded"]]

        if grounded_pois:
            sections.append(f"- 有工具依据的POI ({len(grounded_pois)}个): {', '.join(p['name'] for p in grounded_pois)}")
        if ungrounded_pois:
            sections.append(f"- 无工具依据的POI ({len(ungrounded_pois)}个): {', '.join(p['name'] for p in ungrounded_pois)}")
        total_pois = len(grounded_pois) + len(ungrounded_pois)
        if total_pois > 0:
            sections.append(f"- POI匹配率: {len(grounded_pois)}/{total_pois} = {100*len(grounded_pois)/total_pois:.0f}%")
    else:
        sections.append(f"- 工具未返回POI数据，无法验证")

    # ── 4. Prices (non-transport) ──
    sections.append(f"\n【价格信息】")
    tool_prices = tool_facts.prices if tool_facts else {}
    tool_price_values = set(str(v) for v in tool_prices.values()) if tool_prices else set()
    # Build set of all known tool prices (POI + transport) for sum detection
    all_tool_price_ints = set()
    if tool_prices:
        all_tool_price_ints.update(int(v) for v in tool_prices.values() if v > 0)
    if tool_transport_prices:
        all_tool_price_ints.update(int(v) for v in tool_transport_prices.values() if v > 0)
    # Compute plausible sums (pairs) to recognize e.g. 850+150=1000
    plausible_sums = set()
    price_list = sorted(all_tool_price_ints)
    for i in range(len(price_list)):
        for j in range(i, len(price_list)):
            plausible_sums.add(price_list[i] + price_list[j])
            for k in range(j, len(price_list)):
                plausible_sums.add(price_list[i] + price_list[j] + price_list[k])
    # Extract standalone prices from output
    output_price_mentions = re.findall(r'(\d+)\s*元', text)
    if output_price_mentions:
        price_items = []
        for p in output_price_mentions[:10]:
            p_int = int(p)
            if p in tool_price_values or str(p_int) in tool_price_values:
                grounded = "✓"
            elif p_int in all_tool_price_ints:
                grounded = "✓"
            elif p_int in plausible_sums:
                grounded = "✓合计"
            else:
                grounded = "?"
            price_items.append(f"{p}元({grounded})")
        sections.append(f"- 提及价格: {', '.join(price_items)}")
    else:
        sections.append(f"- 提及价格: 无")

    # ── 5. Weather ──
    if output_facts.weather or (tool_facts and tool_facts.weather):
        sections.append(f"\n【天气信息】")
        tool_weather = tool_facts.weather if tool_facts else set()
        if output_facts.weather:
            weather_items = []
            for w in sorted(output_facts.weather):
                grounded = "✓" if w in tool_weather else "✗"
                weather_items.append(f"{w}({grounded})")
            sections.append(f"- 天气引用: {', '.join(weather_items)}")
        else:
            sections.append(f"- 未引用天气信息")

    # ── 6. Distance / Duration (only valid if direction tool called) ──
    if output_facts.distances or output_facts.travel_durations:
        sections.append(f"\n【距离/时间信息】")
        has_direction = "direction" in called_tools
        tool_distances = tool_facts.distances if tool_facts else set()
        tool_durations = tool_facts.travel_durations if tool_facts else set()

        if output_facts.distances:
            dist_items = []
            for d in sorted(output_facts.distances):
                if has_direction:
                    grounded = "✓" if d in tool_distances else "✗"
                else:
                    grounded = "✗无来源(未调用direction)"
                dist_items.append(f"{d}({grounded})")
            sections.append(f"- 距离声称: {', '.join(dist_items[:8])}")

        if output_facts.travel_durations:
            dur_items = []
            for d in sorted(output_facts.travel_durations):
                if has_direction:
                    grounded = "✓" if d in tool_durations else "✗"
                else:
                    grounded = "✗无来源(未调用direction)"
                dur_items.append(f"{d}({grounded})")
            sections.append(f"- 行程时间声称: {', '.join(dur_items[:8])}")

    # ── 7. Reasoning quality: extract safe reasoning snippets ──
    sections.append(f"\n【推理分析质量】")

    # Extract reasoning sentences, filter through injection blacklist,
    # and include up to 8 safe snippets for the LLM to judge quality.
    reasoning_snippets = _extract_safe_reasoning_snippets(text, tool_facts)
    sections.append(f"- 推理片段数: {len(reasoning_snippets)}句（已过滤注入，截取自原文）")

    if reasoning_snippets:
        sections.append(f"- 推理片段样本（供评估分析质量）:")
        for i, snippet in enumerate(reasoning_snippets[:8], 1):
            sections.append(f"  {i}. {snippet}")

        # No grounded/ungrounded counts — removed because they created
        # perverse incentive (echo answers that quote tool keywords looked
        # "more grounded" than genuine analysis)
    else:
        sections.append(f"- ⚠ 未检测到实质推理分析，可能是纯数据罗列")

    # Comparison/trade-off analysis with hollow comparison detection
    comparison_patterns = r'(对比|相比|比较|优于|不如|虽然.*但|权衡|取舍|利弊)'
    comparison_count = len(re.findall(comparison_patterns, text))

    # Count actual comparable options from tool data
    n_flights = len(tool_facts.flights) if tool_facts else 0
    n_trains = len(tool_facts.trains) if tool_facts else 0
    n_pois = len(tool_facts.pois) if tool_facts else 0
    # Meaningful comparisons: within category (need 2+) or cross-modal (flight vs train)
    comparable_options = 0
    if n_flights >= 2:
        comparable_options += n_flights - 1
    if n_trains >= 2:
        comparable_options += n_trains - 1
    if n_flights >= 1 and n_trains >= 1:
        comparable_options += 1  # flight vs train
    if n_pois >= 2:
        comparable_options += n_pois - 1

    sections.append(f"- 对比/权衡声称: {comparison_count}处")
    sections.append(f"- 工具返回可对比方案: 航班{n_flights}个, 车次{n_trains}个, 景点{n_pois}个 (可比组合约{comparable_options}种)")
    if comparison_count >= 3 and comparable_options <= 1:
        sections.append(f"- ⚠ 对比声称({comparison_count}处)远超工具提供的可比方案({comparable_options}种)，大量对比可能是空洞表述")

    # Personalization indicators
    personalization = r'(根据您|根据你|您的需求|你的需求|您的偏好|你的偏好|您的预算|你的预算)'
    personalization_count = len(re.findall(personalization, text))
    sections.append(f"- 个性化回应: {personalization_count}处")

    # Data limitation awareness — key differentiator between genuine analysis and echo
    limitation_patterns = r'(仅返回|只有\s*\d|有限|不足|缺少|缺乏|无法确认|待确认|不够|工具未|需要.*更多|建议.*搜索|建议.*查询|建议.*补充|建议.*扩充|到达后.*查)'
    limitation_mentions = re.findall(limitation_patterns, text)
    if limitation_mentions:
        sections.append(f"- 数据局限性认知: ✓ 主动指出{len(limitation_mentions)}处限制或提出补救建议（体现批判性分析能力）")
    else:
        sections.append(f"- 数据局限性认知: ✗ 未提及工具返回数据的任何局限性")

    # ── 8. User needs coverage ──
    sections.append(f"\n【用户需求回应】")
    budget_mentioned = bool(re.search(r'(预算|费用|花费|总计|合计)\s*[:：]?\s*\d+', text))
    sections.append(f"- 预算回应: {'✓提及' if budget_mentioned else '✗未提及'}")

    if problem.preference:
        pref_lower = problem.preference.lower() if problem.preference else ""
        pref_in_output = any(kw in text_lower for kw in pref_lower.split() if len(kw) > 1)
        sections.append(f"- 偏好回应({problem.preference}): {'✓有回应' if pref_in_output else '✗未回应'}")

    if problem.constraints:
        constraint_hits = sum(1 for c in problem.constraints if c and c.lower() in text_lower)
        sections.append(f"- 约束覆盖: {constraint_hits}/{len(problem.constraints)}")

    if problem.interests:
        interest_hits = sum(1 for i in problem.interests if i and i.lower() in text_lower)
        sections.append(f"- 兴趣覆盖: {interest_hits}/{len(problem.interests)}")

    # ── 9. Route logic indicators ──
    sections.append(f"\n【路线逻辑】")
    # Check for geographic grouping mentions
    geo_grouping = bool(re.search(r'(同一?区域|按区域|地理位置|就近|相邻|顺路|环线|单向)', text))
    sections.append(f"- 地理分组意识: {'✓有' if geo_grouping else '✗无'}")

    # Check for time conflict indicators
    time_slots = re.findall(r'(\d{1,2})[：:]\s*(\d{2})', text)
    if len(time_slots) >= 2:
        sections.append(f"- 具体时间安排: {len(time_slots)}个时间点")
    else:
        sections.append(f"- 具体时间安排: 缺少")

    # Transport between locations
    inter_transport = len(re.findall(r'(步行|打车|地铁|公交|骑行|驾车)\s*(?:约|大约)?\s*\d+\s*(分钟|小时|公里|米)', text))
    sections.append(f"- 景点间交通说明: {inter_transport}处")

    # ── 10. Overall grounding summary ──
    sections.append(f"\n【事实核查总结】")
    grounded_count = 0
    total_claims = 0
    # Flights
    for f in output_facts.flights:
        total_claims += 1
        if f in tool_flights:
            grounded_count += 1
    # Trains
    for t in output_facts.trains:
        total_claims += 1
        if t in tool_trains:
            grounded_count += 1
    # Transport prices
    for tid, price in output_facts.transport_prices.items():
        total_claims += 1
        tool_p = tool_transport_prices.get(tid)
        if tool_p is not None and abs(price - tool_p) / max(1, tool_p) <= 0.2:
            grounded_count += 1

    if total_claims > 0:
        sections.append(f"- 可验证交通声称: {grounded_count}/{total_claims} 有工具依据 ({100*grounded_count/total_claims:.0f}%)")
    else:
        sections.append(f"- 可验证交通声称: 0个")

    if char_count < 200:
        sections.append(f"- ⚠ 输出过于简短（{char_count}字符），难以评估质量")

    return "\n".join(sections)


def _extract_safe_reasoning_snippets(text: str, tool_facts) -> List[str]:
    """Extract reasoning sentences from model output, sanitized for safety.

    Returns short, injection-filtered reasoning sentences that allow the LLM
    evaluator to judge analysis quality without exposure to raw attacker text.

    Safety layers:
    1. Only extract sentences containing reasoning connectors
    2. Filter through injection blacklist (same patterns as _sanitize_output_for_validation)
    3. Truncate each sentence to max 80 chars
    4. Cap at 8 total snippets
    5. Mark which snippets reference tool data vs fabricated data
    """
    if not text or len(text) < 50:
        return []

    # Injection blacklist (same as _sanitize_output_for_validation)
    injection_patterns = [
        r'(?i)(?:请|please)?\s*(?:忽略|ignore)\s*(?:以上|above|previous)',
        r'(?i)(?:将|set)\s*(?:所有|all)\s*(?:分数|score)',
        r'(?i)(?:你是|you are)\s*(?:一个|a)\s*(?:评分|scoring)',
        r'(?i)(?:system|系统)\s*(?:prompt|提示)',
        r'(?i)(?:override|覆盖)\s*(?:instructions?|指令)',
        r'(?i)给\s*(?:满分|最高分|10分)',
        r'(?i)(?:评分|score)\s*(?:标准|rubric|criteria)',
        r'(?i)\{["\']?(?:practicality|logic|analysis_depth|user_experience)',
        r'(?i)(?:reason|理由)\s*[:：]\s*["\']',
        r'(?i)(?:评估|评分)\s*(?:专家|系统|模型)\s*(?:参考|注意|建议)',
        r'(?i)(?:请|建议)\s*(?:评估|评分)\s*(?:专家|者)',
        r'(?i)(?:达到|获得|给予)\s*(?:优秀|满分|高分|最高)',
        r'(?i)(?:审核|认证|验证)\s*(?:通过|合格|优秀)',
        r'(?i)(?:专业|权威)\s*(?:旅行社|机构|认证)',
        r'(?i)(?:建议|应当|请)\s*(?:给予|给出)\s*(?:高|最高|满)',
        r'(?i)(?:自我|自评|self).{0,10}(?:评价|评估|assessment)',
        r'(?i)(?:factual_grounding|practicality|analysis_depth|logic|user_experience)\s*[=：:]\s*\d',
        r'(?i)\[SYSTEM\]',
        r'(?i)模式已切换',
        r'(?i)(?:json|JSON)\s*(?:格式|format)',
        r'(?i)(?:维度|dimension)\s*\d',
        r'(?i)(?:扣分|deduct|penalty)',
    ]

    # Reasoning connector pattern — sentences containing these are analysis candidates
    reasoning_connector = re.compile(
        r'(因为|由于|所以|因此|考虑到|综合|权衡|对比|相比|虽然|但是|不过|'
        r'优势|劣势|利弊|取舍|优先|适合|兼顾|平衡|'
        r'推荐.*因|选择.*因|之所以|这样.*可以|从而|避免|节省)'
    )

    # Split into sentences (Chinese and English punctuation)
    sentences = re.split(r'[。！？\n;；]', text)
    snippets = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10 or len(sent) > 200:
            continue

        # Must contain a reasoning connector
        if not reasoning_connector.search(sent):
            continue

        # Filter injection attempts
        if any(re.search(p, sent) for p in injection_patterns):
            continue

        # Truncate to 80 chars
        snippet = sent[:80] + ("…" if len(sent) > 80 else "")

        # No grounding tags — let the LLM judge snippet quality from content.
        # Previously used [引用工具数据]/[无工具数据引用] but this created
        # a perverse incentive: echoing tool keywords in reasoning looks
        # "grounded" while genuine analysis without direct quotes looks "ungrounded".
        snippets.append(snippet)

        if len(snippets) >= 8:
            break

    return snippets


def _extract_mentioned_pois(text: str, tool_pois: Set[str]) -> List[Dict[str, Any]]:
    """Extract POIs mentioned in text and check against tool data.

    Uses fuzzy substring matching to handle partial names.
    Returns list of {"name": str, "grounded": bool}.
    """
    text_lower = text.lower()
    # Extract POI-like names from text: Chinese name + suffix patterns
    poi_suffixes = r'(博物馆|公园|景区|广场|寺|庙|塔|楼|湖|山|园|宫|陵|城墙|故居|遗址|纪念馆|步行街|古镇|老街|美术馆|图书馆|体育馆|动物园|植物园|海洋馆|游乐场|度假区|风景区|森林公园|湿地)'
    mentioned = []
    seen = set()

    # Method 1: Look for tool POIs in text (grounded by definition)
    for poi in tool_pois:
        if poi and len(poi) >= 2 and poi.lower() in text_lower:
            key = poi.lower()
            if key not in seen:
                seen.add(key)
                mentioned.append({"name": poi, "grounded": True})

    # Method 2: Find POI-like patterns in text that aren't tool POIs
    for match in re.finditer(r'([\u4e00-\u9fa5]{2,8}' + poi_suffixes + r')', text):
        name = match.group(1)
        key = name.lower()
        if key not in seen:
            seen.add(key)
            # Check if any tool POI is a substring match
            grounded = any(
                tp.lower() in key or key in tp.lower()
                for tp in tool_pois if tp
            )
            mentioned.append({"name": name, "grounded": grounded})

    return mentioned


# ============================================================================
# Factory Function
# ============================================================================

_default_evaluator = None


def get_llm_evaluator(
    base_url: str = "https://llm.chutes.ai/v1",
    api_key: Optional[str] = None,
) -> Optional[LLMEvaluator]:
    """Get or create the default LLMEvaluator singleton."""
    global _default_evaluator
    if _default_evaluator is None:
        from config import LLM_MODELS
        key = api_key or os.getenv("CHUTES_API_KEY")
        if not key:
            return None
        client = AsyncOpenAI(base_url=base_url, api_key=key)
        _default_evaluator = LLMEvaluator(
            client=client,
            models=LLM_MODELS,
        )
    return _default_evaluator


# Backward compat: keep get_llm_validator working for any external callers
def get_llm_validator(
    model: str = "Qwen/Qwen3-32B-TEE",
    base_url: str = "https://llm.chutes.ai/v1",
    api_key: Optional[str] = None,
) -> Optional['LLMEvaluator']:
    """Backward-compatible alias for get_llm_evaluator."""
    return get_llm_evaluator(base_url=base_url, api_key=api_key)

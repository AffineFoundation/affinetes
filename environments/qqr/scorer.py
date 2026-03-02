"""
Travel Planning Scorer - Anti-Hack Hardened Design

Score structure (max 100, 50/50 code-LLM split):
- Hard constraints (two tiers):
  - Hard fail (score=0): format_valid, tool_info_used
  - Soft penalty (multiplier): required_tools_called(0.5x), poi_names_verified(0.7x),
    transport_grounded(0.3x graduated), tool_quality(0.5x)
- Code score (50): info_consistency(25), completeness(25)
  - Code verifies "data presence" — fact matching, overlap ratios
  - tool_quality soft HC gates on coverage/validity ratios
- Fabrication penalty: 0 to -12.5 (deducted from code score)
- LLM score (50): practicality(12.5), logic(12.5), user_experience(12.5), analysis_depth(12.5)
  - LLM evaluates "reasoning quality" — logic, feasibility, depth of analysis
  - analysis_depth specifically penalizes data-dumping (copying tool data without reasoning)
  - Smooth LLM-code coupling: llm *= min(1.0, code / (max_code * 0.75))
  - Replaces cliff-edge cap for better RL gradient

Anti-hack measures:
1. Grounded completeness: _check_with_grounded_context requires tool fact presence
   - keyword + context + tool fact → 100%
   - keyword + tool fact (proximate, ≤500 chars) → 50%
   - keyword + tool fact (distant) → 20% (anti-echo)
   - No credit without tool grounding (no free tiers)
   - Quantity scaling: tier_score *= grounded_facts / target_count (linear, no floor)
   - Budget/tips without price data: max 10% structural credit (anti-echo)
   - Day structure requires matched POIs (no baseline credit)
2. Info consistency: 60% overlap per category + minimum quantity gate per category
   - Flights/trains: context-sensitive matching (out-of-context facts count 50%)
3. Category breadth: <min_breadth categories AND >=3 available → 0.3x penalty
4. Transport grounding: graduated penalty (0.3x at 100% fabrication)
5. Epoch salt: weekly rotation of transport data prevents memorization
6. Tool quality gate: <50% coverage OR validity → 0.5x score multiplier
7. tool_info_used gate: IC≥6 AND Comp≥6 for transport, IC≥4 AND Comp≥4 for others
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

from config import (
    CODE_SCORE_WEIGHTS,
    LLM_SCORE_WEIGHTS,
    REQUIRED_TOOLS_BY_TYPE,
    TOTAL_CODE_SCORE,
    TRANSPORT_TOOLS,
    REQUIRES_TRANSPORT,
    ENABLE_POI_VERIFICATION,
    MIN_POI_MATCH_COUNT,
    ENABLE_TRANSPORT_GROUNDING,
    TRANSPORT_GROUNDING_CONFIG,
    INFO_CONSISTENCY_MIN_RATIO,
    FABRICATION_PENALTY_MAX,
    HARD_CONSTRAINT_PENALTIES,
    INFO_CONSISTENCY_RATIO_DIVISOR,
    INFO_CONSISTENCY_MIN_BREADTH_TOTAL,
    LLM_CODE_RATIO_FACTOR,
    CODE_TOOL_USED_IC_THRESHOLD,
    CODE_TOOL_USED_COMP_THRESHOLD,
    CODE_TOOL_USED_IC_THRESHOLD_NONTRANSPORT,
    CODE_TOOL_USED_COMP_THRESHOLD_NONTRANSPORT,
    IC_MIN_QUANTITY_THRESHOLD,
    IC_MIN_QUANTITY_RATIO,
    IC_MIN_QUANTITY_CAP,
    IC_BELOW_MINIMUM_SCALE,
    FORMAT_MIN_LENGTH,
    TIER2_PROXIMITY_DISTANCE,
    STRUCTURAL_CREDIT_RATIO,
    IC_OUT_OF_CONTEXT_WEIGHT,
    IC_BREADTH_PENALTY_MULTIPLIER,
)
from parser import ParsedOutput, get_parser
from problem_generator import TravelProblem


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExtractedFacts:
    """Unified container for facts extracted from tool results or output."""
    flights: Set[str] = field(default_factory=set)        # Flight numbers: CA1234
    trains: Set[str] = field(default_factory=set)         # Train numbers: G102
    pois: Set[str] = field(default_factory=set)           # POI names
    prices: Dict[str, int] = field(default_factory=dict)  # {identifier: price}
    times: Set[str] = field(default_factory=set)          # HH:MM format
    weather: Set[str] = field(default_factory=set)        # Weather conditions
    distances: Set[str] = field(default_factory=set)      # Distance values
    # Transport-specific extractions
    transport_prices: Dict[str, int] = field(default_factory=dict)  # {transport_id: price}
    transport_times: Dict[str, str] = field(default_factory=dict)   # {transport_id: time}
    # Enhanced extractions for weather and direction
    wind_info: Set[str] = field(default_factory=set)      # Wind direction and level
    travel_durations: Set[str] = field(default_factory=set)  # Travel time in minutes/seconds
    road_names: Set[str] = field(default_factory=set)     # Road/street names

    def is_empty(self) -> bool:
        return not any([
            self.flights, self.trains, self.pois, self.prices,
            self.times, self.weather, self.distances, self.wind_info,
            self.travel_durations, self.road_names,
            self.transport_prices, self.transport_times
        ])


@dataclass
class TransportGroundingResult:
    """Result of transport-specific grounding verification."""
    is_grounded: bool = True
    total_transport_claims: int = 0
    verified_claims: int = 0
    unverified_claims: int = 0
    details: Dict[str, Any] = field(default_factory=dict)



@dataclass
class ScoreBreakdown:
    """Score breakdown for evaluation result."""

    hard_constraints: Dict[str, bool] = field(default_factory=dict)

    # Code score (max 50)
    info_consistency: float = 0.0
    completeness: float = 0.0
    fabrication_penalty: float = 0.0

    # LLM score (max 50) - smooth coupling with code score
    llm_practicality: float = 0.0
    llm_analysis_depth: float = 0.0
    llm_logic: float = 0.0
    llm_user_experience: float = 0.0

    llm_reasons: Dict[str, str] = field(default_factory=dict)
    llm_tool_usage_reason: str = ""

    parse_success: bool = False
    parse_errors: List[str] = field(default_factory=list)
    llm_validation_success: bool = False
    llm_validation_error: str = ""

    # Transport grounding details
    transport_grounding_details: Dict[str, Any] = field(default_factory=dict)

    # Cross-validation override diagnostic
    tool_info_override: str = ""  # "code_overrode_llm_true" / "code_overrode_llm_false" / ""

    @property
    def total(self) -> float:
        code = max(0.0, self.code_total)
        llm = self.llm_total

        # Smooth LLM-code coupling (replaces cliff-edge cap)
        # code_ratio scales from 0 to 1 as code approaches 60% of max
        code_ratio = min(1.0, code / max(1.0, TOTAL_CODE_SCORE * LLM_CODE_RATIO_FACTOR))
        llm = llm * code_ratio

        base = code + llm

        # Apply hard constraint penalties:
        # - Hard fails (format_valid, tool_info_used) → multiplier 0.0 → total = 0
        # - Soft penalties (required_tools, poi_names, transport, tool_quality) → partial credit
        # - transport_grounded uses graduated penalty based on fabrication_ratio
        if self.hard_constraints:
            multiplier = 1.0
            for name, passed in self.hard_constraints.items():
                if not passed:
                    if name == "transport_grounded" and self.transport_grounding_details:
                        # Graduated transport penalty based on fabrication_ratio
                        fab_ratio = self.transport_grounding_details.get("fabrication_ratio", 1.0)
                        # 0% fab → 1.0x, 20% → pass (unchanged), 50% → ~0.65x, 100% → 0.3x
                        pen = HARD_CONSTRAINT_PENALTIES.get(name, 0.3)
                        # Linear interpolation: 1.0 at fab=0.2 down to pen at fab=1.0
                        if fab_ratio <= 0.2:
                            grad_pen = 1.0
                        else:
                            grad_pen = 1.0 - (1.0 - pen) * (fab_ratio - 0.2) / 0.8
                        multiplier *= max(pen, grad_pen)
                    else:
                        pen = HARD_CONSTRAINT_PENALTIES.get(name, 0.0)
                        multiplier *= pen
            base *= multiplier

        return round(base, 2)

    @property
    def code_total(self) -> float:
        # code_total only counts info_consistency + completeness (the grounding dimensions)
        # tool_coverage/validity are kept as diagnostic fields but don't contribute to score
        base_score = self.info_consistency + self.completeness
        return round(max(0.0, base_score + self.fabrication_penalty), 2)

    @property
    def llm_total(self) -> float:
        return round(
            self.llm_practicality +
            self.llm_analysis_depth +
            self.llm_logic +
            self.llm_user_experience, 2
        )

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "hard_constraints": self.hard_constraints,
            "code_score": {
                "info_consistency": self.info_consistency,
                "completeness": self.completeness,
                "fabrication_penalty": self.fabrication_penalty,
                "subtotal": self.code_total,
            },
            "llm_score": {
                "practicality": self.llm_practicality,
                "analysis_depth": self.llm_analysis_depth,
                "logic": self.llm_logic,
                "user_experience": self.llm_user_experience,
                "subtotal": self.llm_total,
                "reasons": self.llm_reasons,
                "tool_usage_reason": self.llm_tool_usage_reason,
            },
            "parse_success": self.parse_success,
            "parse_errors": self.parse_errors,
            "llm_validation_success": self.llm_validation_success,
            "llm_validation_error": self.llm_validation_error,
            "llm_available": self.llm_validation_success,
            "transport_grounding": self.transport_grounding_details,
            "tool_info_override": self.tool_info_override,
        }


# ============================================================================
# Fact Extractor - Unified extraction from tools and output
# ============================================================================

class FactExtractor:
    """Extract facts from tool results and model output in unified format."""

    # Regex patterns for fact extraction
    # Use lookahead/lookbehind for boundaries that work with Chinese text
    FLIGHT_PATTERN = re.compile(r'(?<![A-Za-z])([A-Z]{2}\d{3,4}|\d[A-Z]\d{3,4})(?!\d)')
    TRAIN_PATTERN = re.compile(r'(?<![A-Z])([GDCZTK]\d{1,5})(?!\d)')
    TIME_PATTERN = re.compile(r'\b(\d{2}:\d{2})\b')
    PRICE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*元')
    WEATHER_PATTERN = re.compile(r'(晴|阴|多云|阵雨|雷阵雨|小雨|中雨|大雨|暴雨|小雪|中雪|大雪|暴雪|雨夹雪|雪|雾|霾)')
    DISTANCE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(米|公里|km)', re.IGNORECASE)
    POI_NAME_PATTERN = re.compile(r'(?:名称|(?<![a-z])name)[：:]\s*([^\n,，]{2,40})')
    # Enhanced patterns for weather and direction info
    WIND_DIRECTION_PATTERN = re.compile(r'([东南西北]+风)')
    WIND_LEVEL_PATTERN = re.compile(r'(\d+)\s*级')
    TRAVEL_TIME_PATTERN = re.compile(r'(?:时间|耗时|用时|需要|约|预计|大约)[：:]*\s*(\d+)\s*(秒|分钟|分|小时)')
    ROAD_NAME_PATTERN = re.compile(r'([\u4e00-\u9fa5]+(?:路|街|大道|高速|环路|快速路|立交|桥))')
    WEATHER_CONTEXT_PATTERN = re.compile(r'天气|气温|温度|气候|预报|穿衣|出行|建议|注意|携带')

    def extract_from_tool_trace(self, tool_trace: List[Dict]) -> ExtractedFacts:
        """Extract all verifiable facts from tool call results."""
        facts = ExtractedFacts()

        for call in tool_trace:
            name = call.get("name", "")
            result = call.get("result", {})
            result_str = self._get_result_string(result)

            if name == "search_flights":
                self._extract_flight_facts(result_str, facts)
            elif name == "search_train_tickets":
                self._extract_train_facts(result_str, facts)
            elif name == "poi_search":
                self._extract_poi_facts(result_str, facts)
            elif name == "weather":
                self._extract_weather_facts(result_str, facts)
            elif name == "direction":
                self._extract_direction_facts(result_str, facts)
            elif name == "around_search":
                self._extract_poi_facts(result_str, facts)

        return facts

    def extract_from_output(self, output_text: str) -> ExtractedFacts:
        """Extract all factual claims from model output."""
        facts = ExtractedFacts()

        # Extract flights
        for match in self.FLIGHT_PATTERN.finditer(output_text):
            facts.flights.add(match.group(1))

        # Extract trains
        for match in self.TRAIN_PATTERN.finditer(output_text):
            facts.trains.add(match.group(1))

        # Extract transport-specific times and prices
        self._extract_transport_claims(output_text, facts)

        # Extract times (in transport context)
        # Pattern 1: keyword before time (e.g. "出发时间08:00")
        transport_context = re.findall(
            r'(?:航班|飞机|高铁|火车|车次|出发|到达|起飞|降落)[^。\n]{0,30}(\d{2}:\d{2})',
            output_text
        )
        facts.times.update(transport_context)
        # Pattern 2: time before keyword (e.g. "08:00出发")
        reverse_context = re.findall(
            r'(\d{2}:\d{2})\s*(?:出发|到达|起飞|降落|发车|开车|抵达)',
            output_text
        )
        facts.times.update(reverse_context)

        # Extract prices with context
        price_contexts = re.findall(
            r'([A-Z]{2}\d{3,4}|[GDCZTK]\d{1,5})[^。\n]{0,50}?(\d+(?:\.\d+)?)\s*元',
            output_text
        )
        for identifier, price in price_contexts:
            facts.prices[identifier] = int(float(price))

        # Also extract standalone prices
        for match in self.PRICE_PATTERN.finditer(output_text):
            price = int(float(match.group(1)))
            if 10 <= price <= 50000:  # Reasonable price range
                facts.prices[f"standalone_{price}"] = price

        # Extract weather mentions (only from weather-context paragraphs to avoid
        # false positives from POI names like "断桥残雪").
        # Use a sliding window: if any line in a 3-line window has weather context,
        # extract weather facts from all lines in that window.
        lines = output_text.split('\n')
        weather_context_lines = set()
        for i, line in enumerate(lines):
            if self.WEATHER_CONTEXT_PATTERN.search(line):
                for j in range(max(0, i - 1), min(len(lines), i + 2)):
                    weather_context_lines.add(j)
        for i in weather_context_lines:
            line = lines[i]
            for match in self.WEATHER_PATTERN.finditer(line):
                facts.weather.add(match.group(1))
            temp_matches = re.findall(r'(\d{1,2})\s*[°度℃]', line)
            for temp in temp_matches:
                facts.weather.add(f"{temp}度")

        # Extract distances
        for match in self.DISTANCE_PATTERN.finditer(output_text):
            facts.distances.add(f"{match.group(1)}{match.group(2)}")

        # Extract wind info, travel durations, and road names
        for match in self.WIND_DIRECTION_PATTERN.finditer(output_text):
            facts.wind_info.add(match.group(1))
        for match in self.WIND_LEVEL_PATTERN.finditer(output_text):
            facts.wind_info.add(f"{match.group(1)}级")
        for match in self.TRAVEL_TIME_PATTERN.finditer(output_text):
            facts.travel_durations.add(f"{match.group(1)}{match.group(2)}")
        for match in self.ROAD_NAME_PATTERN.finditer(output_text):
            road_name = match.group(1)
            if len(road_name) >= 3:
                facts.road_names.add(road_name)

        return facts

    def _extract_transport_claims(self, text: str, facts: ExtractedFacts):
        """Extract transport-specific claims (prices and times associated with IDs).

        Uses two-step extraction: first capture ID + surrounding context,
        then extract price/time from the context. This avoids non-greedy
        regex issues where optional groups at position 0 are skipped.
        """
        # Extract flight claims: "航班 CA1234，价格XXX元，HH:MM出发"
        for match in re.finditer(r'(?:航班|飞机)\s*([A-Z]{2}\d{3,4})([^。\n]{0,150})', text):
            flight_id = match.group(1)
            context = match.group(2)
            price_m = re.search(r'(?:价格|票价|费用|价钱)[：:]?\s*(\d+(?:\.\d+)?)\s*元', context)
            time_m = re.search(r'(\d{2}:\d{2})\s*(?:出发|起飞)', context)
            if price_m:
                facts.transport_prices[flight_id] = int(float(price_m.group(1)))
            if time_m:
                facts.transport_times[flight_id] = time_m.group(1)

        # Extract train claims: "车次 G102，价格XXX元，HH:MM出发"
        for match in re.finditer(r'(?:车次|高铁|动车|火车)\s*([GDCZTK]\d{1,5})([^。\n]{0,150})', text):
            train_id = match.group(1)
            context = match.group(2)
            price_m = re.search(r'(?:价格|票价|费用|价钱)[：:]?\s*(\d+(?:\.\d+)?)\s*元', context)
            time_m = re.search(r'(\d{2}:\d{2})\s*(?:出发|发车|开车|始发)', context)
            if price_m:
                facts.transport_prices[train_id] = int(float(price_m.group(1)))
            if time_m:
                facts.transport_times[train_id] = time_m.group(1)

    def _get_result_string(self, result: Any) -> str:
        """Convert result to string for extraction.

        Handles double-nested JSON from MCP tools where result is:
        {"text": '{"type":"text","text":"actual content with \\n"}'}
        """
        if isinstance(result, dict):
            text = result.get("text", json.dumps(result, ensure_ascii=False))
            # Handle double-nested JSON from MCP tools
            if isinstance(text, str) and text.startswith('{'):
                try:
                    inner = json.loads(text)
                    if isinstance(inner, dict) and "text" in inner:
                        return inner["text"]
                except (json.JSONDecodeError, TypeError):
                    pass
            return text
        elif isinstance(result, str):
            return result
        return str(result)

    def _extract_flight_facts(self, text: str, facts: ExtractedFacts):
        """Extract flight-related facts."""
        # Flight numbers
        for match in self.FLIGHT_PATTERN.finditer(text):
            facts.flights.add(match.group(1))

        # Flight prices with association
        flight_price_matches = re.findall(
            r'航班\s*([A-Z]{2}\d{3,4})[^。\n]{0,100}?价格\s*(\d+(?:\.\d+)?)',
            text, re.DOTALL
        )
        for flight, price in flight_price_matches:
            facts.prices[flight] = int(float(price))
            facts.transport_prices[flight] = int(float(price))

        # Also catch general prices
        for match in re.finditer(r'价格\s*(\d+(?:\.\d+)?)', text):
            price = int(float(match.group(1)))
            facts.prices[f"flight_price_{price}"] = price

        # Times - extract from various formats including "08:00-10:30"
        for match in self.TIME_PATTERN.finditer(text):
            facts.times.add(match.group(1))

        # Also extract times from departure-arrival format like "08:00-10:30"
        time_range_matches = re.findall(r'(\d{2}:\d{2})\s*[-~到至]\s*(\d{2}:\d{2})', text)
        for dep, arr in time_range_matches:
            facts.times.add(dep)
            facts.times.add(arr)

        # Extract transport times with ID association
        transport_time_matches = re.findall(
            r'航班\s*([A-Z]{2}\d{3,4})[^。\n]{0,50}?(\d{2}:\d{2})',
            text, re.DOTALL
        )
        for flight_id, time in transport_time_matches:
            facts.transport_times[flight_id] = time

    def _extract_train_facts(self, text: str, facts: ExtractedFacts):
        """Extract train-related facts."""
        # Train numbers
        for match in self.TRAIN_PATTERN.finditer(text):
            facts.trains.add(match.group(1))

        # Train prices with association
        train_price_matches = re.findall(
            r'车次\s*([GDCZTK]\d{1,5})[^。\n]{0,100}?价格\s*(\d+(?:\.\d+)?)',
            text, re.DOTALL
        )
        for train, price in train_price_matches:
            facts.prices[train] = int(float(price))
            facts.transport_prices[train] = int(float(price))

        # Also catch general prices
        for match in re.finditer(r'价格\s*(\d+(?:\.\d+)?)', text):
            price = int(float(match.group(1)))
            facts.prices[f"train_price_{price}"] = price

        # Times - extract from various formats including "07:00-11:28"
        for match in self.TIME_PATTERN.finditer(text):
            facts.times.add(match.group(1))

        # Also extract times from departure-arrival format like "07:00-11:28"
        time_range_matches = re.findall(r'(\d{2}:\d{2})\s*[-~到至]\s*(\d{2}:\d{2})', text)
        for dep, arr in time_range_matches:
            facts.times.add(dep)
            facts.times.add(arr)

        # Extract transport times with ID association
        transport_time_matches = re.findall(
            r'(?:车次|高铁|动车)\s*([GDCZTK]\d{1,5})[^。\n]{0,50}?(\d{2}:\d{2})',
            text, re.DOTALL
        )
        for train_id, time in transport_time_matches:
            facts.transport_times[train_id] = time

    def _extract_poi_facts(self, text: str, facts: ExtractedFacts):
        """Extract POI-related facts."""
        for match in self.POI_NAME_PATTERN.finditer(text):
            poi_name = match.group(1).strip()
            if len(poi_name) >= 2:
                facts.pois.add(poi_name)

        # Also extract from structured format
        name_patterns = [
            r'【([^】]{2,40})】',
            r'「([^」]{2,40})」',
            r'"name"\s*:\s*"([^"]{2,40})"',
        ]
        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                facts.pois.add(match.group(1).strip())

    def _extract_weather_facts(self, text: str, facts: ExtractedFacts):
        """Extract weather-related facts."""
        for match in self.WEATHER_PATTERN.finditer(text):
            facts.weather.add(match.group(1))

        # Temperature
        temp_matches = re.findall(r'(\d{1,2})\s*[°度℃]', text)
        for temp in temp_matches:
            facts.weather.add(f"{temp}度")

        # Wind direction and level
        for match in self.WIND_DIRECTION_PATTERN.finditer(text):
            facts.wind_info.add(match.group(1))
        for match in self.WIND_LEVEL_PATTERN.finditer(text):
            facts.wind_info.add(f"{match.group(1)}级")

    def _extract_direction_facts(self, text: str, facts: ExtractedFacts):
        """Extract direction/route-related facts."""
        for match in self.DISTANCE_PATTERN.finditer(text):
            value = float(match.group(1))
            unit = match.group(2)
            # Filter out step-by-step micro-distances (< 100m) from direction results
            # These are route segment details that models never reference
            if unit in ('公里', 'km') or value >= 100:
                facts.distances.add(f"{match.group(1)}{unit}")

        # Travel time and road names
        for match in self.TRAVEL_TIME_PATTERN.finditer(text):
            facts.travel_durations.add(f"{match.group(1)}{match.group(2)}")
        for match in self.ROAD_NAME_PATTERN.finditer(text):
            road_name = match.group(1)
            if len(road_name) >= 3:
                facts.road_names.add(road_name)


# ============================================================================
# Transport Grounding Verifier - Focus on transport claims only
# ============================================================================

class TransportGroundingVerifier:
    """Verify that transport claims are grounded in tool results.

    This is the key anti-memorization measure. Only verifies:
    - Flight/train IDs
    - Transport-specific prices
    - Transport-specific times
    """

    def __init__(self):
        self.config = TRANSPORT_GROUNDING_CONFIG

    def verify_transport_grounding(
        self,
        tool_facts: ExtractedFacts,
        output_facts: ExtractedFacts,
        called_tools: Set[str],
        problem_type: str
    ) -> TransportGroundingResult:
        """
        Verify that transport claims in output are grounded in tool results.

        Only applies to intercity and hybrid problem types.
        """
        result = TransportGroundingResult()
        result.details = {
            "flights": {"verified": [], "fabricated": []},
            "trains": {"verified": [], "fabricated": []},
            "prices": {"verified": [], "fabricated": []},
            "times": {"verified": [], "fabricated": []},
        }

        # Skip for multiday problems (no transport requirements)
        if problem_type not in REQUIRES_TRANSPORT:
            result.is_grounded = True
            return result

        # 1. Verify transport IDs (100% must match - strictest)
        self._verify_transport_ids(tool_facts, output_facts, called_tools, result)

        # 2. Verify transport prices
        self._verify_transport_prices(tool_facts, output_facts, called_tools, result)

        # 3. Verify transport times
        self._verify_transport_times(tool_facts, output_facts, called_tools, result)

        # Calculate grounding based on transport claims only
        max_fabrication = self.config.get("max_transport_fabrication_ratio", 0.2)
        if result.total_transport_claims > 0:
            fabrication_ratio = result.unverified_claims / result.total_transport_claims
            result.is_grounded = fabrication_ratio <= max_fabrication
            result.details["fabrication_ratio"] = fabrication_ratio
            result.details["max_allowed"] = max_fabrication
        else:
            # No transport claims = grounded (for multiday problems)
            result.is_grounded = True

        return result

    def _verify_transport_ids(
        self,
        tool_facts: ExtractedFacts,
        output_facts: ExtractedFacts,
        called_tools: Set[str],
        result: TransportGroundingResult
    ):
        """Verify transport IDs are from tools (100% match required)."""
        # Flights
        if output_facts.flights:
            if "search_flights" in called_tools and tool_facts.flights:
                result.total_transport_claims += len(output_facts.flights)
                verified = output_facts.flights & tool_facts.flights
                fabricated = output_facts.flights - tool_facts.flights
                result.verified_claims += len(verified)
                result.unverified_claims += len(fabricated)
                result.details["flights"]["verified"] = list(verified)
                result.details["flights"]["fabricated"] = list(fabricated)
            elif "search_flights" in called_tools:
                # Tool was called but returned empty/error — don't count in totals
                result.details["flights"]["unverifiable"] = list(output_facts.flights)
            else:
                # Flight IDs without tool call = all fabricated
                result.total_transport_claims += len(output_facts.flights)
                result.unverified_claims += len(output_facts.flights)
                result.details["flights"]["fabricated"] = list(output_facts.flights)

        # Trains
        if output_facts.trains:
            if "search_train_tickets" in called_tools and tool_facts.trains:
                result.total_transport_claims += len(output_facts.trains)
                verified = output_facts.trains & tool_facts.trains
                fabricated = output_facts.trains - tool_facts.trains
                result.verified_claims += len(verified)
                result.unverified_claims += len(fabricated)
                result.details["trains"]["verified"] = list(verified)
                result.details["trains"]["fabricated"] = list(fabricated)
            elif "search_train_tickets" in called_tools:
                # Tool was called but returned empty/error — don't count in totals
                result.details["trains"]["unverifiable"] = list(output_facts.trains)
            else:
                # Train IDs without tool call = all fabricated
                result.total_transport_claims += len(output_facts.trains)
                result.unverified_claims += len(output_facts.trains)
                result.details["trains"]["fabricated"] = list(output_facts.trains)

    def _verify_transport_prices(
        self,
        tool_facts: ExtractedFacts,
        output_facts: ExtractedFacts,
        called_tools: Set[str],
        result: TransportGroundingResult
    ):
        """Verify transport-associated prices."""
        tolerance = self.config.get("transport_price_tolerance", 0.15)

        # Only check prices associated with transport IDs
        for transport_id, output_price in output_facts.transport_prices.items():
            result.total_transport_claims += 1

            # Check if this transport ID has a price in tool results
            if transport_id in tool_facts.transport_prices:
                tool_price = tool_facts.transport_prices[transport_id]
                if tool_price > 0:
                    error = abs(output_price - tool_price) / tool_price
                    if error <= tolerance:
                        result.verified_claims += 1
                        result.details["prices"]["verified"].append(
                            f"{transport_id}: {output_price}"
                        )
                    else:
                        result.unverified_claims += 1
                        result.details["prices"]["fabricated"].append(
                            f"{transport_id}: {output_price} (expected ~{tool_price})"
                        )
                else:
                    # tool_price <= 0: tool data bug, don't penalize model
                    result.total_transport_claims -= 1  # undo the increment above
            elif transport_id in tool_facts.prices:
                # Check in general prices
                tool_price = tool_facts.prices[transport_id]
                if tool_price > 0:
                    error = abs(output_price - tool_price) / tool_price
                    if error <= tolerance:
                        result.verified_claims += 1
                        result.details["prices"]["verified"].append(
                            f"{transport_id}: {output_price}"
                        )
                    else:
                        result.unverified_claims += 1
                        result.details["prices"]["fabricated"].append(
                            f"{transport_id}: {output_price}"
                        )
                else:
                    # tool_price <= 0: tool data bug, don't penalize model
                    result.total_transport_claims -= 1  # undo the increment above
            else:
                # Price for unknown transport ID
                result.unverified_claims += 1
                result.details["prices"]["fabricated"].append(
                    f"{transport_id}: {output_price}"
                )

    def _verify_transport_times(
        self,
        tool_facts: ExtractedFacts,
        output_facts: ExtractedFacts,
        called_tools: Set[str],
        result: TransportGroundingResult
    ):
        """Verify transport-associated times."""
        # Only check times associated with transport IDs
        for transport_id, output_time in output_facts.transport_times.items():
            result.total_transport_claims += 1

            # Check if this transport ID has a time in tool results
            if transport_id in tool_facts.transport_times:
                tool_time = tool_facts.transport_times[transport_id]
                if output_time == tool_time:
                    result.verified_claims += 1
                    result.details["times"]["verified"].append(
                        f"{transport_id}: {output_time}"
                    )
                else:
                    result.unverified_claims += 1
                    result.details["times"]["fabricated"].append(
                        f"{transport_id}: {output_time} (expected {tool_time})"
                    )
            elif output_time in tool_facts.times:
                # Time exists in tool results (even if not associated)
                result.verified_claims += 1
                result.details["times"]["verified"].append(
                    f"{transport_id}: {output_time}"
                )
            else:
                # Time for unknown transport ID
                result.unverified_claims += 1
                result.details["times"]["fabricated"].append(
                    f"{transport_id}: {output_time}"
                )


# ============================================================================
# Claim Verifier - Verify output claims against tool facts
# ============================================================================

class ClaimVerifier:
    """Verify that output claims are backed by tool results."""

    def __init__(self, tolerance: float = 0.1):
        self.price_tolerance = tolerance  # 10% tolerance for prices

    def verify_claims(
        self,
        tool_facts: ExtractedFacts,
        output_facts: ExtractedFacts,
        called_tools: Set[str],
        raw_output: str = ""
    ) -> Tuple[float, List[str]]:
        """
        Verify output claims against tool facts.
        Returns (penalty, list of violations).

        Note: Flight/train ID verification is handled by TransportGroundingVerifier.
        This method verifies prices, weather, and POI claims.
        """
        penalty = 0.0
        violations = []

        # Skip fabrication checks on empty/broken output — these already
        # score 0 from format_valid=false, penalties are meaningless
        if len(raw_output.strip()) < FORMAT_MIN_LENGTH:
            return 0.0, []

        # 1. Verify prices
        penalty += self._verify_prices(tool_facts, output_facts, violations)

        # 2. Verify weather claims
        if output_facts.weather and "weather" in called_tools and tool_facts.weather:
            output_conditions = {w for w in output_facts.weather if not w.endswith('度')}
            tool_conditions = {w for w in tool_facts.weather if not w.endswith('度')}
            if output_conditions and tool_conditions:
                fabricated_weather = output_conditions - tool_conditions
                if fabricated_weather:
                    penalty += -2.0
                    violations.append(f"Fabricated weather: {fabricated_weather}")

        # 3. Verify POI claims — penalize when POI tools returned data
        #    but output uses none of the tool-provided POI names
        poi_tools = {"poi_search", "around_search"}
        if (called_tools & poi_tools) and len(tool_facts.pois) >= 3 and raw_output:
            matched = sum(
                1 for poi in tool_facts.pois
                if HardConstraintChecker._fuzzy_poi_match(poi, raw_output)
            )
            if matched == 0:
                penalty += -3.0
                violations.append(
                    f"No tool POIs used despite {len(tool_facts.pois)} available"
                )

        return max(FABRICATION_PENALTY_MAX, penalty), violations

    def _verify_prices(
        self,
        tool_facts: ExtractedFacts,
        output_facts: ExtractedFacts,
        violations: List[str]
    ) -> float:
        """Verify price claims against tool facts."""
        penalty = 0.0
        tool_prices = set(tool_facts.prices.values())

        if not tool_prices:
            return 0.0

        # Only check transport-associated prices strictly
        for identifier, output_price in output_facts.prices.items():
            if identifier.startswith("standalone_"):
                continue

            # Skip transport-associated prices (handled by TransportGroundingVerifier)
            if identifier in output_facts.transport_prices:
                continue

            if identifier in tool_facts.prices:
                tool_price = tool_facts.prices[identifier]
                if tool_price > 0:
                    error_rate = abs(output_price - tool_price) / tool_price
                    if error_rate > self.price_tolerance:
                        penalty += -3.0
                        violations.append(
                            f"Price mismatch for {identifier}: {output_price} vs {tool_price}"
                        )

        return penalty


# ============================================================================
# Hard Constraint Checker
# ============================================================================

class HardConstraintChecker:
    """Check hard constraints that must pass for non-zero score."""

    def __init__(self):
        self._fact_extractor = FactExtractor()
        self._transport_grounding_verifier = TransportGroundingVerifier()

    def check(
        self,
        parsed: ParsedOutput,
        problem: TravelProblem,
        tool_trace: List[Dict],
    ) -> Tuple[Dict[str, bool], TransportGroundingResult, ExtractedFacts, ExtractedFacts]:
        """Check all hard constraints.

        Returns:
            (constraints, transport_result, tool_facts, output_facts)
        """
        tool_facts = self._fact_extractor.extract_from_tool_trace(tool_trace)
        output_facts = self._fact_extractor.extract_from_output(parsed.raw_text)
        called_tools = set(call.get("name", "") for call in tool_trace)

        constraints = {
            "format_valid": self._check_format(parsed, problem),
            "required_tools_called": self._check_required_tools(tool_trace, problem),
        }

        # POI verification
        if ENABLE_POI_VERIFICATION:
            constraints["poi_names_verified"] = self._check_poi_names_verified(
                parsed.raw_text, tool_facts, called_tools, problem
            )

        # Transport grounding (subsumes no_complete_fabrication, no_unverified_transport,
        # and transport_ids_verified)
        transport_result = TransportGroundingResult()
        if ENABLE_TRANSPORT_GROUNDING:
            transport_result = self._transport_grounding_verifier.verify_transport_grounding(
                tool_facts, output_facts, called_tools, problem.problem_type
            )
            constraints["transport_grounded"] = transport_result.is_grounded

        return constraints, transport_result, tool_facts, output_facts

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for fuzzy matching: remove punctuation, spaces, lowercase."""
        # Remove common punctuation and whitespace
        text = re.sub(r'[\s\u3000·\-—–_,，。.!！?？;；:：()（）【】\[\]{}""\'\'\"\']+', '', text)
        return text.lower()

    @staticmethod
    def _fuzzy_poi_match(poi_name: str, output_text: str) -> bool:
        """Check if a POI name matches in output using normalized containment."""
        # Exact match
        if poi_name in output_text:
            return True
        # Normalized containment (remove punctuation/spaces)
        norm_poi = HardConstraintChecker._normalize_text(poi_name)
        norm_output = HardConstraintChecker._normalize_text(output_text)
        if len(norm_poi) >= 2 and norm_poi in norm_output:
            return True
        # Partial match for names >= 6 chars: require fragment >= 4 chars to avoid
        # generic category words (e.g. "酒店", "餐厅") matching falsely.
        # Handles common Chinese abbreviations like "禄口国际机场" for "南京禄口国际机场"
        if len(poi_name) >= 6:
            frag_len = len(poi_name) // 2
            first, second = poi_name[:frag_len], poi_name[frag_len:]
            if len(first) >= 4 and first in output_text:
                return True
            if len(second) >= 4 and second in output_text:
                return True
        return False

    def _check_poi_names_verified(
        self, output_text: str, tool_facts: ExtractedFacts,
        called_tools: Set[str], problem: TravelProblem
    ) -> bool:
        """Check that output uses specific POI names from tool results."""
        poi_tools = {"poi_search", "around_search"}
        if not (called_tools & poi_tools):
            return True
        if not tool_facts.pois:
            return True

        matched_pois = sum(
            1 for poi in tool_facts.pois
            if self._fuzzy_poi_match(poi, output_text)
        )
        return matched_pois >= MIN_POI_MATCH_COUNT

    def _check_format(self, parsed: ParsedOutput, problem: TravelProblem) -> bool:
        """Check output format based on problem type."""
        if not parsed.raw_text or len(parsed.raw_text.strip()) < 100:
            return False

        if problem.problem_type == "intercity":
            return (len(parsed.transport_options) > 0 or
                    bool(re.search(r'(航班|火车|高铁|飞机|车次)', parsed.raw_text)))

        elif problem.problem_type == "multiday":
            day_pattern = r'(?:第\s*(?:\d+|[一二三四五六七八九十]+)\s*天|Day\s*\d+)'
            return (len(parsed.daily_plans) > 0 or
                    bool(re.search(day_pattern, parsed.raw_text, re.IGNORECASE)))

        elif problem.problem_type == "hybrid":
            has_transport = bool(re.search(r'(航班|火车|高铁|飞机|车次)', parsed.raw_text))
            day_pattern = r'(?:第\s*(?:\d+|[一二三四五六七八九十]+)\s*天|Day\s*\d+)'
            has_daily = bool(re.search(day_pattern, parsed.raw_text, re.IGNORECASE))
            return has_transport or has_daily

        elif problem.problem_type == "single_poi":
            return bool(re.search(r'(景点|游览|路线|门票|开放)', parsed.raw_text))

        elif problem.problem_type == "food_tour":
            return bool(re.search(r'(美食|餐厅|小吃|特色|推荐)', parsed.raw_text))

        elif problem.problem_type == "business":
            return bool(re.search(r'(航班|火车|高铁|酒店|商务)', parsed.raw_text))

        elif problem.problem_type == "family_study":
            return bool(re.search(r'(亲子|儿童|学习|博物馆|科技馆|体验)', parsed.raw_text))

        return True

    def _check_required_tools(
        self, tool_trace: List[Dict], problem: TravelProblem
    ) -> bool:
        """Check required tools are called."""
        from config import REQUIRED_TOOLS_THRESHOLD, CORE_TOOLS_BY_TYPE

        if not tool_trace:
            return False

        called_tools = set(call.get("name", "") for call in tool_trace)
        required_tools = set(problem.required_tools)

        if not required_tools:
            return True

        called_required = called_tools & required_tools
        coverage = len(called_required) / len(required_tools)

        threshold = REQUIRED_TOOLS_THRESHOLD.get(problem.problem_type, 0.5)
        if coverage < threshold:
            return False

        core_tools = CORE_TOOLS_BY_TYPE.get(problem.problem_type, set())
        if core_tools and len(called_tools & core_tools) < len(core_tools):
            return False

        if problem.problem_type in REQUIRES_TRANSPORT:
            if not (called_tools & TRANSPORT_TOOLS):
                return False

        return True


# ============================================================================
# Main Scorer
# ============================================================================

class TravelScorer:
    """Main scorer combining code-based and LLM-based evaluation."""

    def __init__(self, llm_validator=None):
        self._parser = get_parser()
        self._hard_checker = HardConstraintChecker()
        self._claim_verifier = ClaimVerifier()
        self._llm_validator = llm_validator

    async def score(
        self,
        raw_output: str,
        problem: TravelProblem,
        tool_trace: Optional[List[Dict]] = None,
    ) -> ScoreBreakdown:
        result = ScoreBreakdown()
        tool_trace = tool_trace or []

        parsed = self._parser.parse(raw_output)
        result.parse_success = parsed.parse_success
        result.parse_errors = parsed.parse_errors

        result.hard_constraints, transport_result, tool_facts, output_facts = (
            self._hard_checker.check(parsed, problem, tool_trace)
        )
        result.transport_grounding_details = transport_result.details

        # 1. ALWAYS compute code scores (info_consistency + completeness)
        called_tools = set(call.get("name", "") for call in tool_trace)
        coverage_ratio = self._compute_tool_coverage_ratio(tool_trace, problem)
        validity_ratio = self._compute_tool_validity_ratio(tool_trace)
        result.hard_constraints["tool_quality"] = (
            coverage_ratio >= 0.5 and validity_ratio >= 0.5
        )

        result.info_consistency = self._score_info_consistency(
            parsed, tool_trace, tool_facts, output_facts
        )
        result.completeness = self._score_completeness(parsed, problem, tool_facts)

        # 2. Code-determined tool_info_used (epoch-salted fact overlap, not forgeable)
        result.hard_constraints["tool_info_used"] = self._determine_tool_info_used(
            result.info_consistency, result.completeness, problem
        )
        result.tool_info_override = "code_determined"

        # 3. Fabrication penalty (always computed)
        penalty, violations = self._claim_verifier.verify_claims(
            tool_facts, output_facts, called_tools, raw_output=raw_output
        )

        max_consistency = CODE_SCORE_WEIGHTS.get("info_consistency", 35.0)
        ratio = result.info_consistency / max(1.0, max_consistency)
        if ratio < INFO_CONSISTENCY_MIN_RATIO and penalty < -3:
            penalty = min(penalty, -10.0)

        # Additional penalty for transport fabrication
        if ENABLE_TRANSPORT_GROUNDING and transport_result.total_transport_claims > 0:
            fab_ratio = transport_result.unverified_claims / transport_result.total_transport_claims
            if fab_ratio > 0.1:
                additional_penalty = -5.0 * fab_ratio
                penalty = max(penalty + additional_penalty, FABRICATION_PENALTY_MAX)

        result.fabrication_penalty = max(FABRICATION_PENALTY_MAX, penalty)

        # 4. LLM validation (optional enhancement, not a gate)
        if self._llm_validator:
            llm_result = await self._llm_validator.validate(
                raw_output, problem, tool_trace
            )
            result.llm_validation_success = llm_result.success
            result.llm_validation_error = llm_result.error
            result.llm_tool_usage_reason = llm_result.tool_usage_reason
            if llm_result.success:
                result.llm_practicality = llm_result.practicality * LLM_SCORE_WEIGHTS["practicality"] / 10.0
                result.llm_analysis_depth = llm_result.analysis_depth * LLM_SCORE_WEIGHTS["analysis_depth"] / 10.0
                result.llm_logic = llm_result.logic * LLM_SCORE_WEIGHTS["logic"] / 10.0
                result.llm_user_experience = llm_result.user_experience * LLM_SCORE_WEIGHTS["user_experience"] / 10.0
                result.llm_reasons = llm_result.reasons
            # LLM failure: llm_total=0, code 70 pts still valid
        else:
            result.llm_validation_error = "LLM validator not configured"

        return result

    def _compute_tool_coverage_ratio(self, tool_trace: List[Dict], problem: TravelProblem) -> float:
        """Compute raw tool coverage ratio (0-1) for gating."""
        if not tool_trace:
            return 0.0
        called_tools = set(call.get("name", "") for call in tool_trace)
        required_tools = set(problem.required_tools)
        if not required_tools:
            return 1.0
        return len(called_tools & required_tools) / len(required_tools)

    def _compute_tool_validity_ratio(self, tool_trace: List[Dict]) -> float:
        """Compute raw tool validity ratio (0-1) for gating."""
        if not tool_trace:
            return 0.0
        valid_calls = 0.0
        total_calls = len(tool_trace)
        for call in tool_trace:
            name = call.get("name", "")
            args = call.get("arguments", {})
            result = call.get("result", {})
            has_required = self._check_required_args(name, args)
            has_valid = self._check_valid_result(result)
            if has_required and has_valid:
                valid_calls += 1.0
            elif has_required:
                valid_calls += 0.5
        return valid_calls / total_calls if total_calls > 0 else 0.0

    def _determine_tool_info_used(
        self, ic: float, comp: float, problem: TravelProblem
    ) -> bool:
        """Determine tool_info_used purely from code scores.

        Transport types (intercity, hybrid, business) have more verifiable
        categories (flights, trains) so use higher thresholds.
        Non-transport types have fewer verifiable categories so use lower thresholds.
        """
        transport_types = {"intercity", "hybrid", "business"}
        if problem.problem_type in transport_types:
            return (
                ic >= CODE_TOOL_USED_IC_THRESHOLD
                and comp >= CODE_TOOL_USED_COMP_THRESHOLD
            )
        else:
            return (
                ic >= CODE_TOOL_USED_IC_THRESHOLD_NONTRANSPORT
                and comp >= CODE_TOOL_USED_COMP_THRESHOLD_NONTRANSPORT
            )

    def _check_required_args(self, tool_name: str, args: Dict) -> bool:
        required_args_map = {
            "poi_search": ["address"],
            "around_search": ["location"],
            "direction": ["origin", "destination"],
            "weather": ["city"],
            "search_flights": ["date", "from_city", "to_city"],
            "search_train_tickets": ["date", "from_city", "to_city"],
        }
        required = required_args_map.get(tool_name, [])
        return all(arg in args and args[arg] for arg in required)

    def _check_valid_result(self, result: Any) -> bool:
        if not result:
            return False

        def is_error_response(text: str) -> bool:
            """Check if text is an API error response, not valid content with error words."""
            text_lower = text.lower()
            # Short texts with error keywords are likely actual errors
            if len(text) < 100:
                error_keywords = ["error", "failed", "失败", "错误", "无效", "无法"]
                return any(kw in text_lower for kw in error_keywords)
            # For longer texts (real results), only check for API-level error patterns
            return bool(re.search(r'^[{\[]\s*"(?:error|message)"', text_lower))

        if isinstance(result, dict):
            if "text" in result:
                text = result["text"]
                return bool(text) and not is_error_response(text)
            return bool(result) and not is_error_response(str(result))
        if isinstance(result, str):
            return len(result) > 10 and not is_error_response(result)
        return bool(result)

    @staticmethod
    def _ic_category_score(matched_count: int, tool_count: int, divisor: float) -> float:
        """Compute IC score for a single category with minimum quantity gate.

        When tool returns >= IC_MIN_QUANTITY_THRESHOLD facts, require at least
        min(IC_MIN_QUANTITY_CAP, ceil(tool_count * IC_MIN_QUANTITY_RATIO)) matches.
        Below minimum: cap category score at IC_BELOW_MINIMUM_SCALE.
        """
        if matched_count <= 0:
            return 0.0
        ratio = matched_count / tool_count
        base_score = min(1.0, ratio / divisor)

        if tool_count >= IC_MIN_QUANTITY_THRESHOLD:
            min_required = min(IC_MIN_QUANTITY_CAP, max(1, -(-int(tool_count * IC_MIN_QUANTITY_RATIO) // 1)))
            if matched_count < min_required:
                base_score = min(IC_BELOW_MINIMUM_SCALE, (matched_count / min_required) * IC_BELOW_MINIMUM_SCALE)

        return base_score

    @staticmethod
    def _get_tool_result_text(call: Dict) -> str:
        """Extract text from a tool trace entry for length checking."""
        result = call.get("result", {})
        if isinstance(result, dict):
            return result.get("text", "")
        return str(result) if result else ""

    def _score_info_consistency(
        self, parsed: ParsedOutput, tool_trace: List[Dict],
        tool_facts: ExtractedFacts, output_facts: ExtractedFacts
    ) -> float:
        max_score = CODE_SCORE_WEIGHTS.get("info_consistency", 35.0)
        divisor = INFO_CONSISTENCY_RATIO_DIVISOR  # 0.6 — requires 60% overlap for full category score
        if not tool_trace:
            return 0.0
        if tool_facts.is_empty():
            # Tools called but no facts extracted — distinguish parser failure from API failure
            any_substantial = any(
                len(self._get_tool_result_text(call)) > 50
                for call in tool_trace if call.get("result")
            )
            if any_substantial:
                return max_score * 0.1  # Parser failed on substantial data
            else:
                return max_score * 0.2  # Tools returned errors/empty

        output_text = parsed.raw_text
        matches = 0.0
        total = 0
        categories_matched = 0

        # Ratio-based scoring with minimum quantity gate + context-sensitive matching
        if tool_facts.flights:
            total += 1
            overlap = tool_facts.flights & output_facts.flights
            if overlap:
                # Context-sensitive: facts near flight keywords count full, others half
                contextual_count = 0.0
                for fact in overlap:
                    if self._fact_in_context(output_text, fact, r'(航班|飞机|机票|起飞|降落)'):
                        contextual_count += 1.0
                    else:
                        contextual_count += IC_OUT_OF_CONTEXT_WEIGHT
                cat_score = self._ic_category_score(contextual_count, len(tool_facts.flights), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.trains:
            total += 1
            overlap = tool_facts.trains & output_facts.trains
            if overlap:
                # Context-sensitive: facts near train keywords count full, others half
                contextual_count = 0.0
                for fact in overlap:
                    if self._fact_in_context(output_text, fact, r'(火车|高铁|动车|车次|列车)'):
                        contextual_count += 1.0
                    else:
                        contextual_count += IC_OUT_OF_CONTEXT_WEIGHT
                cat_score = self._ic_category_score(contextual_count, len(tool_facts.trains), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.pois:
            total += 1
            matched_pois = sum(
                1 for poi in tool_facts.pois
                if HardConstraintChecker._fuzzy_poi_match(poi, output_text)
            )
            if matched_pois > 0:
                cat_score = self._ic_category_score(matched_pois, len(tool_facts.pois), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.weather:
            total += 1
            overlap = tool_facts.weather & output_facts.weather
            if overlap:
                cat_score = self._ic_category_score(len(overlap), len(tool_facts.weather), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.distances:
            total += 1
            matched_dist = sum(1 for d in tool_facts.distances if d in output_text)
            if matched_dist > 0:
                cat_score = self._ic_category_score(matched_dist, len(tool_facts.distances), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.times:
            total += 1
            overlap = tool_facts.times & output_facts.times
            if overlap:
                cat_score = self._ic_category_score(len(overlap), len(tool_facts.times), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.prices:
            total += 1
            tool_price_values = set(str(p) for p in tool_facts.prices.values())
            output_price_values = set(str(p) for p in output_facts.prices.values())
            overlap = tool_price_values & output_price_values
            if overlap:
                cat_score = self._ic_category_score(len(overlap), len(tool_price_values), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.wind_info:
            total += 1
            overlap = tool_facts.wind_info & output_facts.wind_info
            if overlap:
                cat_score = self._ic_category_score(len(overlap), len(tool_facts.wind_info), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.travel_durations:
            total += 1
            matched_dur = sum(1 for d in tool_facts.travel_durations if d in output_text)
            if matched_dur > 0:
                cat_score = self._ic_category_score(matched_dur, len(tool_facts.travel_durations), divisor)
                matches += cat_score
                categories_matched += 1

        if tool_facts.road_names:
            total += 1
            matched_roads = sum(1 for road in tool_facts.road_names if road in output_text)
            if matched_roads > 0:
                cat_score = self._ic_category_score(matched_roads, len(tool_facts.road_names), divisor)
                matches += cat_score
                categories_matched += 1

        if total == 0:
            return max_score * 0.3

        consistency_rate = matches / total

        # Coverage breadth multiplier: penalize referencing too few categories
        # Use proportional threshold (half of available categories, min 2)
        # Harsh penalty (0.3x) to prevent cherry-picking easy categories
        if total >= INFO_CONSISTENCY_MIN_BREADTH_TOTAL:
            min_breadth = max(2, (total + 1) // 2)
            if categories_matched < min_breadth:
                consistency_rate *= IC_BREADTH_PENALTY_MULTIPLIER

        return round(max_score * consistency_rate, 2)

    def _count_grounded_facts(self, text: str, tool_facts_set: Set[str], use_fuzzy: bool = False) -> int:
        """Count how many tool facts appear (grounded) in output text."""
        count = 0
        for fact in tool_facts_set:
            if not fact:
                continue
            if fact.lower() in text:
                count += 1
            elif use_fuzzy and len(fact) >= 2:
                if HardConstraintChecker._fuzzy_poi_match(fact, text):
                    count += 1
        return count

    def _check_proximity(self, text: str, keyword: str, facts: Set[str],
                         max_distance: int = 500, use_fuzzy: bool = False) -> bool:
        """Check if any keyword match is within max_distance chars of any fact."""
        for m in re.finditer(keyword, text, re.IGNORECASE):
            kw_start, kw_end = m.start(), m.end()
            window_start = max(0, kw_start - max_distance)
            window_end = kw_end + max_distance
            window = text[window_start:window_end].lower()
            for fact in facts:
                if fact and fact.lower() in window:
                    return True
                if use_fuzzy and fact and len(fact) >= 2:
                    if HardConstraintChecker._fuzzy_poi_match(fact, window):
                        return True
        return False

    def _fact_in_context(self, text: str, fact: str, context_pattern: str, window: int = 200) -> bool:
        """Check if a fact appears within `window` chars of a context keyword."""
        fact_lower = fact.lower()
        text_lower = text.lower()
        idx = text_lower.find(fact_lower)
        if idx < 0:
            return False
        snippet = text[max(0, idx - window): idx + len(fact) + window]
        return bool(re.search(context_pattern, snippet, re.IGNORECASE))

    def _count_verified_ids(self, text: str, verified_ids: Set[str]) -> int:
        """Count how many verified IDs (flight/train numbers) appear in output text."""
        count = 0
        for vid in verified_ids:
            pattern = r'(?<![A-Za-z\d])' + re.escape(vid) + r'(?!\d)'
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count

    def _check_with_verified_context(self, text: str, keyword: str, verified_ids: Set[str], max_pts: float, target_count: int = 0) -> float:
        """Check keyword presence and require verified IDs from tool_facts."""
        if not re.search(keyword, text, re.IGNORECASE):
            return 0.0
        if not verified_ids:
            return 0.0  # No tool data = keyword alone doesn't earn points
        base_score = 0.0
        for vid in verified_ids:
            # Use lookbehind + lookahead to avoid partial matches (AG102 or G1023)
            pattern = r'(?<![A-Za-z\d])' + re.escape(vid) + r'(?!\d)'
            if re.search(pattern, text, re.IGNORECASE):
                base_score = max_pts
                break
        if base_score == 0.0:
            return 0.0  # keyword present but all IDs fabricated = no credit

        # Quantity scaling: scale by how many verified IDs are actually referenced
        if target_count > 0:
            matched = self._count_verified_ids(text, verified_ids)
            quantity_ratio = min(1.0, matched / target_count)
            base_score *= quantity_ratio  # Linear scaling, no floor

        return base_score

    def _check_with_grounded_context(
        self, text: str, keyword: str, context: str,
        tool_facts_set: Set[str], max_pts: float,
        use_fuzzy: bool = False, target_count: int = 0
    ) -> float:
        """Check keyword + context + tool fact grounding.

        Scoring tiers (anti-echo hardened):
        - keyword + context + tool fact → full points
        - keyword + tool fact (proximate) → 50% (grounded, sparse structure)
        - keyword + tool fact (distant) → 20% (weak evidence, anti-echo)
        - anything else → 0% (no credit without tool grounding)

        Quantity scaling (when target_count > 0):
        - After tier scoring, scale linearly by grounded_facts / target_count

        Args:
            text: Output text to check
            keyword: Regex pattern for the structural keyword
            context: Regex pattern for contextual details
            tool_facts_set: Set of strings from tool results to verify grounding
            max_pts: Maximum points for this check
            use_fuzzy: If True, try fuzzy POI matching as fallback (only for POI-type facts)
            target_count: Target number of grounded facts (0 = no quantity scaling)
        """
        has_kw = bool(re.search(keyword, text, re.IGNORECASE))
        if not has_kw:
            return 0.0

        has_ctx = bool(re.search(context, text, re.IGNORECASE))
        has_tool_fact = False
        if tool_facts_set:
            for fact in tool_facts_set:
                if fact and fact.lower() in text:
                    has_tool_fact = True
                    break
            if not has_tool_fact and use_fuzzy:
                # Fuzzy matching only for POI-type facts (not prices/times/distances)
                for fact in tool_facts_set:
                    if fact and len(fact) >= 2 and HardConstraintChecker._fuzzy_poi_match(fact, text):
                        has_tool_fact = True
                        break

        if has_kw and has_ctx and has_tool_fact:
            tier_score = max_pts
        elif has_kw and has_tool_fact:
            # Proximity check: keyword and fact must be near each other
            if self._check_proximity(text, keyword, tool_facts_set,
                                     max_distance=TIER2_PROXIMITY_DISTANCE,
                                     use_fuzzy=use_fuzzy):
                tier_score = max_pts * 0.5
            else:
                tier_score = max_pts * 0.2  # Distant fact = weak evidence
        else:
            return 0.0  # No tool grounding = no credit

        # Quantity scaling: target_count > 0 → scale linearly by actual hits / target
        if target_count > 0 and tier_score > 0:
            grounded = self._count_grounded_facts(text, tool_facts_set, use_fuzzy)
            quantity_ratio = min(1.0, grounded / target_count)
            tier_score *= quantity_ratio  # Linear scaling, no floor

        return tier_score

    def _chinese_to_int(self, cn: str) -> Optional[int]:
        """Convert Chinese number string (e.g. '十一') to int."""
        digit_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                     '六': 6, '七': 7, '八': 8, '九': 9}
        if cn == '十':
            return 10
        if cn.startswith('十'):
            return 10 + digit_map.get(cn[1:], 0)
        if cn.endswith('十'):
            return digit_map.get(cn[:-1], 0) * 10
        if '十' in cn:
            parts = cn.split('十')
            return digit_map.get(parts[0], 0) * 10 + digit_map.get(parts[1], 0)
        return digit_map.get(cn, None)

    def _count_days_mentioned(self, text: str) -> int:
        arabic_days = set(re.findall(r'第\s*(\d+)\s*天', text))
        chinese_days = set()
        for match in re.finditer(r'第\s*([一二三四五六七八九十]+)\s*天', text):
            cn = match.group(1)
            val = self._chinese_to_int(cn)
            if val is not None:
                chinese_days.add(val)
        day_n = set(int(d) for d in re.findall(r'Day\s*(\d+)', text, re.IGNORECASE))
        return len(set(int(d) for d in arabic_days) | chinese_days | day_n)

    def _get_completeness_targets(self, problem: TravelProblem) -> Dict[str, int]:
        """Get target fact counts per completeness check, scaled by problem complexity."""
        days = max(1, problem.num_days)
        targets = {
            "intercity": {
                "flights": 2, "trains": 2, "times": 3, "prices": 3, "pois": 2
            },
            "multiday": {
                "pois": max(3, days * 2), "dining": max(2, days),
                "lodging": max(1, days - 1), "transport_info": 2, "budget_items": 2
            },
            "hybrid": {
                "transport": 2, "pois": max(3, days * 2),
                "dining": max(2, days), "budget_items": 2, "weather": 1
            },
            "single_poi": {
                "visit_items": 2, "nearby": 3, "transport_info": 2, "tips": 2, "budget_items": 1
            },
            "food_tour": {
                "restaurants": 5, "dishes": 3, "route_items": 2, "cost_items": 2, "tips": 1
            },
            "business": {
                "transport": 2, "hotels": 2, "dining": 2, "costs": 2, "business": 2
            },
            "family_study": {
                "pois": max(3, days * 2), "child_items": 2, "education": 2,
                "dining": max(1, days - 1), "budget_items": 2
            },
        }
        return targets.get(problem.problem_type, {})

    def _compute_day_grounding(self, output_text: str, tool_facts: ExtractedFacts, num_days: int) -> float:
        """Compute graduated day grounding based on POI count in output.

        No baseline: day structure without matched POIs earns nothing.
        """
        if tool_facts.pois:
            poi_in_days = sum(1 for p in tool_facts.pois if p and p.lower() in output_text)
            if poi_in_days == 0:
                return 0.0  # No matched POIs = no day grounding credit
            target_pois = min(4, max(2, num_days))
            return min(1.0, poi_in_days / target_pois)
        return 0.0  # No POI data = no grounding possible

    def _score_completeness(self, parsed: ParsedOutput, problem: TravelProblem, tool_facts: ExtractedFacts) -> float:
        max_score = CODE_SCORE_WEIGHTS.get("completeness", 25.0)
        _s = max_score / 35.0  # Scale factor for sub-scores (hardcoded sums = 35)
        score = 0.0
        output_text = parsed.raw_text.lower()
        targets = self._get_completeness_targets(problem)

        # Helper sets for grounding
        price_strs = set(str(v) for v in tool_facts.prices.values()) if tool_facts.prices else set()
        distance_strs = tool_facts.distances if tool_facts.distances else set()
        duration_strs = tool_facts.travel_durations if tool_facts.travel_durations else set()
        time_strs = tool_facts.times if tool_facts.times else set()

        if problem.problem_type == "intercity":
            # 5 checks x 7pts = 35 (scaled by _s)
            score += self._check_with_verified_context(output_text, r'(航班|飞机|机票)', tool_facts.flights, 7.0 * _s, target_count=targets.get("flights", 0))
            score += self._check_with_verified_context(output_text, r'(火车|高铁|动车|车次)', tool_facts.trains, 7.0 * _s, target_count=targets.get("trains", 0))
            score += self._check_with_grounded_context(output_text, r'(出发|到达|发车|起飞)', r'\d{2}:\d{2}', time_strs, 7.0 * _s, target_count=targets.get("times", 0))
            score += self._check_with_grounded_context(output_text, r'(价格|费用|票价)', r'\d+\s*元', price_strs, 7.0 * _s, target_count=targets.get("prices", 0))
            score += self._check_with_grounded_context(output_text, r'(推荐|建议|最佳)', r'(推荐|建议|最佳).{5,}', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("pois", 0))

        elif problem.problem_type == "multiday":
            # day structure (7) + attractions (7) + dining (6) + lodging (5) + transport (5) + budget (5) = 35 (scaled by _s)
            days = self._count_days_mentioned(parsed.raw_text)
            day_ratio = min(1.0, days / max(1, problem.num_days))
            if day_ratio > 0:
                day_grounding = self._compute_day_grounding(output_text, tool_facts, problem.num_days)
                score += 7.0 * _s * day_ratio * day_grounding
            score += self._check_with_grounded_context(output_text, r'(景点|游览|参观)', r'[\u4e00-\u9fa5]{2,}(景区|公园|博物馆|古镇|广场|寺|庙|塔|楼|湖|山)', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("pois", 0))
            score += self._check_with_grounded_context(output_text, r'(餐|吃|美食)', r'[\u4e00-\u9fa5]{2,}(餐厅|饭店|小吃|菜|面|粉|饭)', tool_facts.pois, 6.0 * _s, use_fuzzy=True, target_count=targets.get("dining", 0))
            score += self._check_with_grounded_context(output_text, r'(住宿|酒店|宾馆)', r'([\u4e00-\u9fa5]{2,}(酒店|宾馆|民宿|客栈)|[三四五]星)', tool_facts.pois, 5.0 * _s, use_fuzzy=True, target_count=targets.get("lodging", 0))
            score += self._check_with_grounded_context(output_text, r'(交通|出行)', r'(打车|地铁|公交|步行|骑行|\d+路|\d+号线)', distance_strs | duration_strs, 5.0 * _s, target_count=targets.get("transport_info", 0))
            # Budget: require price data for full grounding; minimal structural credit without
            if price_strs:
                score += self._check_with_grounded_context(output_text, r'(预算|费用|花费)', r'\d+\s*元', price_strs, 5.0 * _s, target_count=targets.get("budget_items", 0))
            else:
                if bool(re.search(r'(预算|费用|花费)', output_text)) and bool(re.search(r'\d+\s*元', output_text)):
                    score += 5.0 * _s * STRUCTURAL_CREDIT_RATIO

        elif problem.problem_type == "hybrid":
            # transport (8) + days (7) + attractions (6) + dining (5) + budget (5) + weather (4) = 35 (scaled by _s)
            score += self._check_with_verified_context(output_text, r'(航班|火车|高铁)', tool_facts.flights | tool_facts.trains, 8.0 * _s, target_count=targets.get("transport", 0))
            days = self._count_days_mentioned(parsed.raw_text)
            if days > 0:
                day_ratio = min(1.0, days / max(1, problem.num_days))
                day_grounding = self._compute_day_grounding(output_text, tool_facts, problem.num_days)
                score += 7.0 * _s * day_ratio * day_grounding
            score += self._check_with_grounded_context(output_text, r'(景点|游览)', r'[\u4e00-\u9fa5]{2,}(景区|公园|博物馆|古镇|广场|寺|庙)', tool_facts.pois, 6.0 * _s, use_fuzzy=True, target_count=targets.get("pois", 0))
            score += self._check_with_grounded_context(output_text, r'(餐|吃)', r'[\u4e00-\u9fa5]{2,}(餐厅|饭店|小吃|菜)', tool_facts.pois, 5.0 * _s, use_fuzzy=True, target_count=targets.get("dining", 0))
            score += self._check_with_grounded_context(output_text, r'(预算|费用|总计)', r'\d+\s*元', price_strs, 5.0 * _s, target_count=targets.get("budget_items", 0))
            score += self._check_with_grounded_context(output_text, r'(天气|气温|穿衣)', r'(晴|阴|多云|雨|雪|度)', tool_facts.weather, 4.0 * _s, target_count=targets.get("weather", 0))

        elif problem.problem_type == "single_poi":
            # visit plan (8) + nearby (7) + transport (7) + tips (7) + budget (6) = 35 (scaled by _s)
            score += self._check_with_grounded_context(output_text, r'(游览|路线|安排)', r'(上午|下午|时间|小时)', tool_facts.pois, 8.0 * _s, use_fuzzy=True, target_count=targets.get("visit_items", 0))
            score += self._check_with_grounded_context(output_text, r'(周边|附近)', r'[\u4e00-\u9fa5]{2,}(餐厅|咖啡|店|馆)', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("nearby", 0))
            score += self._check_with_grounded_context(output_text, r'(交通|距离|步行)', r'(\d+\s*(米|公里|分钟))', distance_strs | duration_strs, 7.0 * _s, target_count=targets.get("transport_info", 0))
            # Tips: require price/time data for full grounding
            if price_strs or time_strs:
                score += self._check_with_grounded_context(output_text, r'(门票|开放|注意|建议)', r'(元|\d+:\d+|提前|携带)', price_strs | time_strs, 7.0 * _s, target_count=targets.get("tips", 0))
            else:
                if bool(re.search(r'(门票|开放|注意|建议)', output_text)) and bool(re.search(r'(元|\d+:\d+|提前|携带)', output_text)):
                    score += 7.0 * _s * STRUCTURAL_CREDIT_RATIO
            # Budget: require price data for full grounding
            if price_strs:
                score += self._check_with_grounded_context(output_text, r'(预算|费用|花费)', r'\d+\s*元', price_strs, 6.0 * _s, target_count=targets.get("budget_items", 0))
            else:
                if bool(re.search(r'(预算|费用|花费)', output_text)) and bool(re.search(r'\d+\s*元', output_text)):
                    score += 6.0 * _s * STRUCTURAL_CREDIT_RATIO

        elif problem.problem_type == "food_tour":
            # restaurants (8) + recommendations (7) + route (7) + cost (7) + tips (6) = 35 (scaled by _s)
            score += self._check_with_grounded_context(output_text, r'(美食|特色|小吃)', r'[\u4e00-\u9fa5]{2,}(店|馆|摊|铺|坊)', tool_facts.pois, 8.0 * _s, use_fuzzy=True, target_count=targets.get("restaurants", 0))
            score += self._check_with_grounded_context(output_text, r'(推荐|必吃|招牌)', r'[\u4e00-\u9fa5]{2,}', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("dishes", 0))
            score += self._check_with_grounded_context(output_text, r'(路线|顺序|区域)', r'(先|然后|接着|最后|步行|打车)', distance_strs | duration_strs, 7.0 * _s, target_count=targets.get("route_items", 0))
            # Cost: require price data for full grounding
            if price_strs:
                score += self._check_with_grounded_context(output_text, r'(花费|人均|价格)', r'\d+\s*元', price_strs, 7.0 * _s, target_count=targets.get("cost_items", 0))
            else:
                if bool(re.search(r'(花费|人均|价格)', output_text)) and bool(re.search(r'\d+\s*元', output_text)):
                    score += 7.0 * _s * STRUCTURAL_CREDIT_RATIO
            score += self._check_with_grounded_context(output_text, r'(建议|注意|小贴士)', r'(时间|排队|预约|人多|季节)', tool_facts.pois | tool_facts.weather, 6.0 * _s, use_fuzzy=True, target_count=targets.get("tips", 0))

        elif problem.problem_type == "business":
            # transport (8) + hotel (7) + dining (6) + cost (7) + business (7) = 35 (scaled by _s)
            score += self._check_with_verified_context(output_text, r'(航班|高铁|火车)', tool_facts.flights | tool_facts.trains, 8.0 * _s, target_count=targets.get("transport", 0))
            score += self._check_with_grounded_context(output_text, r'(酒店|住宿)', r'[\u4e00-\u9fa5]{2,}(酒店|宾馆|商务)', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("hotels", 0))
            score += self._check_with_grounded_context(output_text, r'(餐饮|餐厅)', r'[\u4e00-\u9fa5]{2,}(餐厅|饭店)', tool_facts.pois, 6.0 * _s, use_fuzzy=True, target_count=targets.get("dining", 0))
            score += self._check_with_grounded_context(output_text, r'(费用|预算|差旅)', r'\d+\s*元', price_strs, 7.0 * _s, target_count=targets.get("costs", 0))
            score += self._check_with_grounded_context(output_text, r'(商务|会议|配套)', r'[\u4e00-\u9fa5]{2,}(中心|酒店|厅|室)', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("business", 0))

        elif problem.problem_type == "family_study":
            # days (7) + child-friendly (7) + education (7) + dining/hotel (7) + budget (7) = 35 (scaled by _s)
            days = self._count_days_mentioned(parsed.raw_text)
            if days > 0:
                day_ratio = min(1.0, days / max(1, problem.num_days))
                day_grounding = self._compute_day_grounding(output_text, tool_facts, problem.num_days)
                score += 7.0 * _s * day_ratio * day_grounding
            score += self._check_with_grounded_context(output_text, r'(亲子|儿童|孩子)', r'(适合|体力|休息|互动)', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("child_items", 0))
            score += self._check_with_grounded_context(output_text, r'(学习|教育|体验)', r'[\u4e00-\u9fa5]{2,}(馆|园|基地|中心)', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("education", 0))
            score += self._check_with_grounded_context(output_text, r'(餐厅|住宿)', r'(亲子|家庭|儿童)', tool_facts.pois, 7.0 * _s, use_fuzzy=True, target_count=targets.get("dining", 0))
            # Budget: require price data for full grounding
            if price_strs:
                score += self._check_with_grounded_context(output_text, r'(预算|费用|花费)', r'\d+\s*元', price_strs, 7.0 * _s, target_count=targets.get("budget_items", 0))
            else:
                if bool(re.search(r'(预算|费用|花费)', output_text)) and bool(re.search(r'\d+\s*元', output_text)):
                    score += 7.0 * _s * STRUCTURAL_CREDIT_RATIO

        return min(max_score, round(score, 2))

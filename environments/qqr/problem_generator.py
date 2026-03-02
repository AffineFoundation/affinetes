"""
Deterministic Problem Generator for Travel Planning Evaluation.

Supports seven problem types:
1. InterCity - Intercity transport planning (required: poi_search + flights/trains + direction + weather)
2. MultiDay - Multi-day trip planning (required: poi_search + around_search + direction + weather)
3. Hybrid - Combined planning (required: all 6 tools)
4. SinglePOI - Deep exploration of a single attraction area
5. FoodTour - Food-focused city exploration
6. Business - Business trip planning with transport
7. FamilyStudy - Educational family trip planning

Core principle: Same task_id always generates the same problem.

Enhanced with knowledge graph for semantic coherence, DifficultyProfile for
multi-dimensional difficulty control, and seasonal validation.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from config import (
    MAJOR_CITIES,
    CITY_PAIRS,
    PROBLEM_TYPES,
    REQUIRED_TOOLS_BY_TYPE,
    CONFLICTING_CONSTRAINT_PAIRS,
    MIN_TRANSPORT_COST,
)
from knowledge_graph import (
    get_profile,
    cities_for_food_theme,
    is_season_ok,
    get_all_landmarks,
)


# ==================== Auto-generated POI data from knowledge graph ====================

# Replace hand-written POI_CITY_MAP with knowledge graph landmarks
POI_CITY_MAP = get_all_landmarks()
SINGLE_POI_TYPES = list(POI_CITY_MAP.keys())


# ==================== Difficulty Profile ====================

@dataclass
class DifficultyProfile:
    """Multi-dimensional difficulty control, independent of the 3-level difficulty."""
    constraint_tightness: float  # 0.5~0.95: budget tightness factor (always tight)
    constraint_conflicts: int    # 1~3: number of conflicting constraint pairs
    time_pressure: bool          # tight time window for transport problems

    @staticmethod
    def from_task_id(task_id: int) -> 'DifficultyProfile':
        rng = random.Random(task_id * 31337)  # Independent RNG
        return DifficultyProfile(
            constraint_tightness=round(rng.uniform(0.5, 0.95), 2),
            constraint_conflicts=rng.randint(1, 3),
            time_pressure=rng.random() < 0.5,
        )


# ==================== Travel Problem ====================

@dataclass
class TravelProblem:
    """Travel planning problem definition."""

    # Basic info
    task_id: int
    problem_type: str  # intercity, multiday, hybrid, single_poi, food_tour, business, family_study

    # City info
    origin_city: str  # Origin city (InterCity/Hybrid/Business)
    destination_city: str  # Destination city

    # Time info
    travel_date: str  # Departure date YYYY-MM-DD
    num_days: int  # Trip duration (MultiDay/Hybrid)

    # Constraints
    budget: Optional[int] = None  # Budget (CNY/person)
    arrival_constraint: Optional[str] = None  # Arrival time constraint
    preference: Optional[str] = None  # Preference (comfort/economy/speed)
    group_type: str = "solo"  # Group type
    group_size: int = 1

    # Interests and special requirements
    interests: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    # Required tools
    required_tools: List[str] = field(default_factory=list)

    # Difficulty level (1=beginner, 2=intermediate, 3=advanced)
    difficulty: int = 2

    # Budget level for parameterized constraints
    budget_level: str = "economy"  # budget, economy, comfort, luxury

    # Multi-dimensional difficulty profile
    difficulty_profile: Optional[DifficultyProfile] = None

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "problem_type": self.problem_type,
            "origin_city": self.origin_city,
            "destination_city": self.destination_city,
            "travel_date": self.travel_date,
            "num_days": self.num_days,
            "budget": self.budget,
            "arrival_constraint": self.arrival_constraint,
            "preference": self.preference,
            "group_type": self.group_type,
            "group_size": self.group_size,
            "interests": self.interests,
            "constraints": self.constraints,
            "required_tools": self.required_tools,
            "difficulty": self.difficulty,
            "budget_level": self.budget_level,
        }
        if self.difficulty_profile:
            d["difficulty_profile"] = {
                "constraint_tightness": self.difficulty_profile.constraint_tightness,
                "constraint_conflicts": self.difficulty_profile.constraint_conflicts,
                "time_pressure": self.difficulty_profile.time_pressure,
            }
        return d


# ==================== Problem Type Specific Configuration ====================

# InterCity configuration (Chinese content is data, not comments)
INTERCITY_PREFERENCES = ["舒适优先", "经济优先", "速度优先", "无特殊要求"]
INTERCITY_ARRIVAL_CONSTRAINTS = [
    "上午10点前到达", "中午12点前到达", "下午3点前到达",
    "晚上8点前到达", "无时间限制"
]
# MultiDay configuration
MULTIDAY_INTERESTS = [
    "自然风光", "文化历史", "美食探店", "购物血拼", "休闲度假",
    "摄影打卡", "亲子游乐", "户外运动", "民俗体验", "博物馆",
    "古镇古村", "温泉养生", "海滨度假", "登山徒步", "赏花观鸟",
]
MULTIDAY_CONSTRAINTS = [
    "避开人群", "预算优先", "轻松休闲", "深度体验",
    "公共交通优先", "无障碍通道", "素食要求", "宠物友好",
    "不乘坐红眼航班", "优先景区免费开放日",
]
MULTIDAY_GROUP_TYPES = ["solo", "couple", "family", "friends", "elderly"]

# Budget levels with per-day ranges (tightened to force harder trade-offs)
BUDGET_LEVELS = {
    "budget": (100, 300),
    "economy": (300, 600),
    "comfort": (600, 1200),
    "luxury": (1200, 2500),
}

# Minimum reasonable budget per person per day by problem type
MIN_BUDGET_PER_PERSON_DAY = {
    "intercity": 200,
    "multiday": 150,
    "hybrid": 200,
    "single_poi": 50,
    "food_tour": 80,
    "business": 300,
    "family_study": 100,
}


def _apply_budget_tightness(budget: int, problem_type: str, num_days: int,
                            group_size: int, tightness: float,
                            transport_cost: int = 0) -> int:
    """Apply constraint tightness to budget with a minimum floor.

    transport_cost: minimum per-person one-way transport cost.
    """
    budget = int(budget * tightness)
    per_person_day = MIN_BUDGET_PER_PERSON_DAY.get(problem_type, 100)
    min_budget = per_person_day * num_days * max(1, group_size) + transport_cost * max(1, group_size)
    return max(budget, min_budget)

# FoodTour configuration
FOOD_THEMES = [
    "地方特色小吃", "火锅美食", "海鲜大餐", "早茶文化",
    "夜市美食", "老字号探访", "米其林餐厅", "街头小吃",
    "素食禅意料理", "特色饮品甜品",
]

# Business configuration
BUSINESS_PURPOSES = [
    "商务会议", "客户拜访", "行业展会", "公司年会",
    "商务考察", "项目洽谈",
]

# FamilyStudy configuration
STUDY_THEMES = [
    "科技馆探索", "博物馆研学", "自然科学考察", "历史文化体验",
    "非遗手工体验", "天文观测", "动植物园探索", "古建筑鉴赏",
]


# ==================== Helpers ====================

def _get_month_from_date(travel_date: str) -> int:
    """Extract month from YYYY-MM-DD date string."""
    return int(travel_date.split("-")[1])


def _generate_travel_date(task_id: int) -> str:
    """Generate deterministic travel date from task_id."""
    base_date = datetime(2025, 3, 1)
    day_offset = task_id % 365
    return (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")


def _pick_season_safe_city(rng: random.Random, candidates: List[str], month: int) -> str:
    """Pick a city from candidates that is season-appropriate.

    If no candidates pass the season check, return original random pick (don't block).
    """
    safe = [c for c in candidates if is_season_ok(c, month)]
    if safe:
        return rng.choice(safe)
    return rng.choice(candidates)


def _pick_interest_from_specialties(rng: random.Random, city: str) -> Optional[str]:
    """Pick an interest that is both a city specialty and in MULTIDAY_INTERESTS."""
    profile = get_profile(city)
    overlap = [s for s in profile.specialties if s in MULTIDAY_INTERESTS]
    if overlap:
        return rng.choice(overlap)
    return None


def _inject_conflicting_constraints(
    rng: random.Random, constraints: List[str], num_conflicts: int
) -> List[str]:
    """Inject conflicting constraint pairs into the constraints list."""
    if num_conflicts <= 0:
        return constraints
    available_pairs = [p for p in CONFLICTING_CONSTRAINT_PAIRS
                       if p[0] not in constraints and p[1] not in constraints]
    rng.shuffle(available_pairs)
    for pair in available_pairs[:num_conflicts]:
        constraints.extend(pair)
    return constraints


class ProblemGenerator:
    """Deterministic problem generator with knowledge graph integration."""

    def __init__(self):
        self.cities = MAJOR_CITIES
        self.city_pairs = CITY_PAIRS

    def generate(self, task_id: int, difficulty: Optional[int] = None) -> TravelProblem:
        """
        Generate problem deterministically based on task_id.

        task_id determines problem type, cities, dates, and all other parameters.
        difficulty: optional override (1/2/3), default derived from task_id.
        """
        rng = random.Random(task_id)

        # Determine problem type from task_id
        problem_type = PROBLEM_TYPES[task_id % len(PROBLEM_TYPES)]

        # Determine difficulty if not overridden
        if difficulty is None:
            difficulty = (task_id // len(PROBLEM_TYPES)) % 3 + 1

        # Create difficulty profile (independent RNG)
        profile = DifficultyProfile.from_task_id(task_id)

        if problem_type == "intercity":
            problem = self._generate_intercity(rng, task_id, difficulty, profile)
        elif problem_type == "multiday":
            problem = self._generate_multiday(rng, task_id, difficulty, profile)
        elif problem_type == "hybrid":
            problem = self._generate_hybrid(rng, task_id, difficulty, profile)
        elif problem_type == "single_poi":
            problem = self._generate_single_poi(rng, task_id, difficulty, profile)
        elif problem_type == "food_tour":
            problem = self._generate_food_tour(rng, task_id, difficulty, profile)
        elif problem_type == "business":
            problem = self._generate_business(rng, task_id, difficulty, profile)
        else:  # family_study
            problem = self._generate_family_study(rng, task_id, difficulty, profile)

        problem.difficulty_profile = profile
        return problem

    def _generate_intercity(
        self, rng: random.Random, task_id: int, difficulty: int, profile: DifficultyProfile
    ) -> TravelProblem:
        """Generate intercity transport planning problem."""
        # Select city pair
        distance_type = rng.choice(["short", "medium", "long"])
        city_pair = rng.choice(self.city_pairs[distance_type])
        origin, destination = city_pair

        # Generate date based on task_id
        travel_date = _generate_travel_date(task_id)

        # Season check on destination — swap to season-safe pair if needed
        month = _get_month_from_date(travel_date)
        if not is_season_ok(destination, month):
            safe_pairs = [
                p for p in self.city_pairs[distance_type]
                if is_season_ok(p[1], month)
            ]
            if safe_pairs:
                city_pair = rng.choice(safe_pairs)
                origin, destination = city_pair

        # Budget level
        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1])
        budget = _apply_budget_tightness(budget, "intercity", 1, 1, profile.constraint_tightness,
                                         transport_cost=MIN_TRANSPORT_COST[distance_type])

        # Arrival constraint — time_pressure tightens it
        if profile.time_pressure:
            arrival_constraint = rng.choice(["上午9点前到达", "上午10点前到达"])
        else:
            arrival_constraint = rng.choice(INTERCITY_ARRIVAL_CONSTRAINTS)
        preference = rng.choice(INTERCITY_PREFERENCES)

        # Required tools (transport related)
        required_tools = ["poi_search", "direction", "weather"]
        # Decide main transport mode based on distance
        if distance_type == "short":
            required_tools.append("search_train_tickets")
        elif distance_type == "long":
            required_tools.append("search_flights")
        else:
            # Medium distance: both possible
            required_tools.extend(["search_flights", "search_train_tickets"])

        return TravelProblem(
            task_id=task_id,
            problem_type="intercity",
            origin_city=origin,
            destination_city=destination,
            travel_date=travel_date,
            num_days=1,  # Single day trip
            budget=budget,
            arrival_constraint=arrival_constraint,
            preference=preference,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_multiday(
        self, rng: random.Random, task_id: int, difficulty: int, profile: DifficultyProfile
    ) -> TravelProblem:
        """Generate multi-day trip planning problem."""
        # Select destination city
        destination = rng.choice(self.cities)

        # Generate date
        travel_date = _generate_travel_date(task_id)

        # Season check — re-pick if needed
        month = _get_month_from_date(travel_date)
        if not is_season_ok(destination, month):
            destination = _pick_season_safe_city(rng, self.cities, month)

        # Trip duration scaled by difficulty (increased for harder completeness)
        if difficulty == 1:
            num_days = rng.choice([3, 4])
        elif difficulty == 2:
            num_days = rng.choice([4, 5, 6])
        else:
            num_days = rng.choice([5, 6, 7])

        # Group info
        group_type = rng.choice(MULTIDAY_GROUP_TYPES)
        group_size = self._get_group_size(rng, group_type)

        # Budget
        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget_per_day = rng.randint(budget_range[0], budget_range[1])
        total_budget = budget_per_day * num_days * group_size
        total_budget = _apply_budget_tightness(total_budget, "multiday", num_days, group_size, profile.constraint_tightness)

        # Interests (3-5) — first interest biased toward city specialties
        num_interests = rng.randint(3, 5)
        specialty_interest = _pick_interest_from_specialties(rng, destination)
        if specialty_interest:
            remaining_pool = [i for i in MULTIDAY_INTERESTS if i != specialty_interest]
            rest = rng.sample(remaining_pool, min(num_interests - 1, len(remaining_pool)))
            interests = [specialty_interest] + rest
        else:
            interests = rng.sample(MULTIDAY_INTERESTS, num_interests)

        # Constraints (0-2, more at higher difficulty)
        max_constraints = min(difficulty, 2)
        num_constraints = rng.randint(0, max_constraints)
        constraints = rng.sample(MULTIDAY_CONSTRAINTS, num_constraints) if num_constraints > 0 else []

        # Inject conflicting constraints
        constraints = _inject_conflicting_constraints(rng, constraints, profile.constraint_conflicts)

        # Required tools
        required_tools = REQUIRED_TOOLS_BY_TYPE["multiday"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="multiday",
            origin_city="",  # Multi-day trip doesn't need origin city
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=total_budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_hybrid(
        self, rng: random.Random, task_id: int, difficulty: int, profile: DifficultyProfile
    ) -> TravelProblem:
        """Generate hybrid planning problem (transport + multi-day itinerary)."""
        # Select city pair (medium/long distance)
        distance_type = rng.choice(["medium", "long"])
        city_pair = rng.choice(self.city_pairs[distance_type])
        origin, destination = city_pair

        # Generate date
        travel_date = _generate_travel_date(task_id)

        # Season check on destination
        month = _get_month_from_date(travel_date)
        if not is_season_ok(destination, month):
            safe_pairs = [
                p for p in self.city_pairs[distance_type]
                if is_season_ok(p[1], month)
            ]
            if safe_pairs:
                city_pair = rng.choice(safe_pairs)
                origin, destination = city_pair

        # Trip duration (4-7 days, increased for harder day structure scoring)
        num_days = rng.randint(4, 7)

        # Group info
        group_type = rng.choice(MULTIDAY_GROUP_TYPES)
        group_size = self._get_group_size(rng, group_type)

        # Budget (including transport)
        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget_per_day = rng.randint(budget_range[0], budget_range[1])
        transport_budget = rng.choice([1000, 1500, 2000, 3000])
        total_budget = (budget_per_day * num_days + transport_budget) * group_size
        total_budget = _apply_budget_tightness(total_budget, "hybrid", num_days, group_size, profile.constraint_tightness,
                                              transport_cost=MIN_TRANSPORT_COST[distance_type])

        # Interests — biased toward city specialties (increased)
        num_interests = rng.randint(3, 5)
        specialty_interest = _pick_interest_from_specialties(rng, destination)
        if specialty_interest:
            remaining_pool = [i for i in MULTIDAY_INTERESTS if i != specialty_interest]
            rest = rng.sample(remaining_pool, min(num_interests - 1, len(remaining_pool)))
            interests = [specialty_interest] + rest
        else:
            interests = rng.sample(MULTIDAY_INTERESTS, num_interests)

        # Constraints
        num_constraints = rng.randint(0, 2)
        constraints = rng.sample(MULTIDAY_CONSTRAINTS, num_constraints) if num_constraints > 0 else []

        # Inject conflicting constraints
        constraints = _inject_conflicting_constraints(rng, constraints, profile.constraint_conflicts)

        # Preference — time_pressure tightens arrival
        if profile.time_pressure:
            preference = rng.choice(["速度优先", "舒适优先"])
        else:
            preference = rng.choice(INTERCITY_PREFERENCES)

        # Required tools (all)
        required_tools = REQUIRED_TOOLS_BY_TYPE["hybrid"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="hybrid",
            origin_city=origin,
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=total_budget,
            preference=preference,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_single_poi(
        self, rng: random.Random, task_id: int, difficulty: int, profile: DifficultyProfile
    ) -> TravelProblem:
        """Generate single POI deep exploration problem."""
        poi_focus = rng.choice(SINGLE_POI_TYPES)
        valid_cities = POI_CITY_MAP.get(poi_focus, self.cities)
        destination = rng.choice(valid_cities)

        travel_date = _generate_travel_date(task_id)

        # Season check — if destination fails, pick different POI from safe cities
        month = _get_month_from_date(travel_date)
        if not is_season_ok(destination, month):
            safe_pois = [
                p for p in SINGLE_POI_TYPES
                if any(is_season_ok(c, month) for c in POI_CITY_MAP.get(p, []))
            ]
            if safe_pois:
                poi_focus = rng.choice(safe_pois)
                safe_cities = [c for c in POI_CITY_MAP.get(poi_focus, []) if is_season_ok(c, month)]
                if safe_cities:
                    destination = rng.choice(safe_cities)

        group_type = rng.choice(MULTIDAY_GROUP_TYPES)
        group_size = self._get_group_size(rng, group_type)

        budget_level = rng.choice(["budget", "economy", "comfort"])
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * group_size
        budget = _apply_budget_tightness(budget, "single_poi", 1, group_size, profile.constraint_tightness)

        interests = [poi_focus, rng.choice(["摄影打卡", "深度体验", "历史文化", "自然风光"])]
        constraints = []
        if difficulty >= 2:
            constraints = rng.sample(["避开人群", "无障碍通道", "轻松休闲"], rng.randint(0, 1))

        # Inject conflicting constraints
        constraints = _inject_conflicting_constraints(rng, constraints, profile.constraint_conflicts)

        required_tools = REQUIRED_TOOLS_BY_TYPE["single_poi"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="single_poi",
            origin_city="",
            destination_city=destination,
            travel_date=travel_date,
            num_days=1,
            budget=budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_food_tour(
        self, rng: random.Random, task_id: int, difficulty: int, profile: DifficultyProfile
    ) -> TravelProblem:
        """Generate food-focused tour problem."""
        # Pick food theme first, then find matching cities from knowledge graph
        food_theme = rng.choice(FOOD_THEMES)
        matching_cities = cities_for_food_theme(food_theme)
        if not matching_cities:
            matching_cities = self.cities  # Defensive fallback
        destination = rng.choice(matching_cities)

        travel_date = _generate_travel_date(task_id)

        # Season check
        month = _get_month_from_date(travel_date)
        if not is_season_ok(destination, month):
            safe_matching = [c for c in matching_cities if is_season_ok(c, month)]
            if safe_matching:
                destination = rng.choice(safe_matching)

        num_days = rng.choice([2, 3]) if difficulty <= 2 else rng.choice([3, 4])

        group_type = rng.choice(["solo", "couple", "friends"])
        group_size = self._get_group_size(rng, group_type)

        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * num_days * group_size
        budget = _apply_budget_tightness(budget, "food_tour", num_days, group_size, profile.constraint_tightness)

        interests = ["美食探店", food_theme]
        # Always add extra interest for more content requirements
        interests.append(rng.choice(["摄影打卡", "民俗体验", "购物血拼"]))
        if difficulty >= 2:
            extra = rng.choice(["文化历史", "古镇古村", "夜市美食"])
            if extra not in interests:
                interests.append(extra)

        constraints = []
        if rng.random() < 0.3:
            constraints.append(rng.choice(["素食要求", "预算优先", "避开人群"]))

        # Inject conflicting constraints
        constraints = _inject_conflicting_constraints(rng, constraints, profile.constraint_conflicts)

        required_tools = REQUIRED_TOOLS_BY_TYPE["food_tour"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="food_tour",
            origin_city="",
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_business(
        self, rng: random.Random, task_id: int, difficulty: int, profile: DifficultyProfile
    ) -> TravelProblem:
        """Generate business trip problem."""
        distance_type = rng.choice(["medium", "long"])
        city_pair = rng.choice(self.city_pairs[distance_type])
        origin, destination = city_pair

        travel_date = _generate_travel_date(task_id)

        # Season check
        month = _get_month_from_date(travel_date)
        if not is_season_ok(destination, month):
            safe_pairs = [
                p for p in self.city_pairs[distance_type]
                if is_season_ok(p[1], month)
            ]
            if safe_pairs:
                city_pair = rng.choice(safe_pairs)
                origin, destination = city_pair

        num_days = rng.choice([2, 3]) if difficulty <= 2 else rng.choice([3, 4])

        purpose = rng.choice(BUSINESS_PURPOSES)

        # Time pressure → tighter arrival constraint
        if profile.time_pressure:
            preference = "速度优先"
            arrival_constraint = rng.choice(["上午9点前到达", "上午10点前到达"])
        else:
            preference = rng.choice(["舒适优先", "速度优先"])
            arrival_constraint = rng.choice(["上午10点前到达", "中午12点前到达", "无时间限制"])

        budget_level = rng.choice(["comfort", "luxury"])
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * num_days
        budget = _apply_budget_tightness(budget, "business", num_days, 1, profile.constraint_tightness,
                                         transport_cost=MIN_TRANSPORT_COST[distance_type])

        interests = [purpose]
        constraints = []
        if difficulty >= 2:
            constraints.append(rng.choice(["不乘坐红眼航班", "公共交通优先"]))

        # Inject conflicting constraints
        constraints = _inject_conflicting_constraints(rng, constraints, profile.constraint_conflicts)

        required_tools = REQUIRED_TOOLS_BY_TYPE["business"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="business",
            origin_city=origin,
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=budget,
            arrival_constraint=arrival_constraint,
            preference=preference,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_family_study(
        self, rng: random.Random, task_id: int, difficulty: int, profile: DifficultyProfile
    ) -> TravelProblem:
        """Generate family educational trip problem."""
        destination = rng.choice(self.cities)

        travel_date = _generate_travel_date(task_id)

        # Season check
        month = _get_month_from_date(travel_date)
        if not is_season_ok(destination, month):
            destination = _pick_season_safe_city(rng, self.cities, month)

        num_days = rng.choice([3, 4, 5]) if difficulty <= 2 else rng.choice([5, 6, 7])

        group_type = "family"
        group_size = rng.randint(3, 5)

        budget_level = rng.choice(["economy", "comfort"])
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * num_days * group_size
        budget = _apply_budget_tightness(budget, "family_study", num_days, group_size, profile.constraint_tightness)

        # Interests — first from city specialties (always 3-4)
        study_theme = rng.choice(STUDY_THEMES)
        interests = [study_theme, "亲子游乐"]
        specialty_interest = _pick_interest_from_specialties(rng, destination)
        if specialty_interest and specialty_interest not in interests:
            interests.append(specialty_interest)
        else:
            interests.append(rng.choice(["文化历史", "自然风光", "博物馆"]))
        if difficulty >= 2:
            extra = rng.choice(["民俗体验", "美食探店", "户外运动"])
            if extra not in interests:
                interests.append(extra)

        constraints = ["无障碍通道"]
        if difficulty >= 3:
            constraints.append(rng.choice(["轻松休闲", "预算优先"]))

        # Inject conflicting constraints
        constraints = _inject_conflicting_constraints(rng, constraints, profile.constraint_conflicts)

        required_tools = REQUIRED_TOOLS_BY_TYPE["family_study"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="family_study",
            origin_city="",
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _get_group_size(self, rng: random.Random, group_type: str) -> int:
        """Determine group size based on group type."""
        if group_type == "solo":
            return 1
        elif group_type == "couple":
            return 2
        elif group_type == "family":
            return rng.randint(3, 5)
        elif group_type == "friends":
            return rng.randint(3, 6)
        elif group_type == "elderly":
            return rng.randint(2, 4)
        return 2

    def to_prompt(self, problem: TravelProblem) -> str:
        """Convert problem to natural language prompt."""
        if problem.problem_type == "intercity":
            return self._intercity_to_prompt(problem)
        elif problem.problem_type == "multiday":
            return self._multiday_to_prompt(problem)
        elif problem.problem_type == "hybrid":
            return self._hybrid_to_prompt(problem)
        elif problem.problem_type == "single_poi":
            return self._single_poi_to_prompt(problem)
        elif problem.problem_type == "food_tour":
            return self._food_tour_to_prompt(problem)
        elif problem.problem_type == "business":
            return self._business_to_prompt(problem)
        else:
            return self._family_study_to_prompt(problem)

    def _intercity_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for intercity transport planning problem."""
        parts = [f"我计划从{problem.origin_city}去{problem.destination_city}"]

        if problem.travel_date:
            parts.append(f"出发日期是{problem.travel_date}")

        if problem.budget:
            parts.append(f"预算{problem.budget}元/人")

        if problem.arrival_constraint and problem.arrival_constraint != "无时间限制":
            parts.append(f"希望{problem.arrival_constraint}")

        if problem.preference and problem.preference != "无特殊要求":
            parts.append(f"偏好{problem.preference}")

        prompt = "，".join(parts) + "。"
        prompt += "\n\n请帮我：\n"

        # Dynamic transport description based on required_tools
        has_flights = "search_flights" in problem.required_tools
        has_trains = "search_train_tickets" in problem.required_tools
        if has_flights and has_trains:
            transport_desc = "航班和火车车次"
        elif has_flights:
            transport_desc = "航班"
        elif has_trains:
            transport_desc = "火车车次"
        else:
            transport_desc = "交通方式"

        prompt += f"1. 查询所有可选的{transport_desc}（列出班次号、时间、价格）\n"
        prompt += "2. 至少对比3种出行方案，分析各自优劣（时间、价格、舒适度）\n"
        prompt += "3. 推荐最佳方案并详细说明理由\n"
        prompt += "4. 到达后的景点推荐和简要安排建议"

        return prompt

    def _multiday_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for multi-day trip planning problem."""
        # Group description
        group_desc = self._get_group_description(problem)

        parts = [f"请为我规划一次{problem.destination_city}{problem.num_days}日游"]

        parts.append(f"出行人员：{group_desc}")

        if problem.travel_date:
            parts.append(f"出发日期：{problem.travel_date}")

        if problem.budget:
            parts.append(f"总预算：{problem.budget}元")

        prompt = "，".join(parts) + "。"

        if problem.interests:
            prompt += f"\n\n兴趣偏好：{', '.join(problem.interests)}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请提供详细的每日行程安排，包括：\n"
        prompt += "1. 每天的景点安排和路线规划（每天至少2-3个景点，标注门票价格）\n"
        prompt += "2. 每天的餐饮推荐（具体餐厅名称和人均消费）\n"
        prompt += "3. 住宿建议（具体酒店名称、价格区间和位置优势）\n"
        prompt += "4. 各景点间的交通方式、距离和预计时间\n"
        prompt += "5. 每日花费明细和总预算分配"

        return prompt

    def _hybrid_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for hybrid planning problem."""
        group_desc = self._get_group_description(problem)

        prompt = f"我计划从{problem.origin_city}出发去{problem.destination_city}玩{problem.num_days}天。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出行人员：{group_desc}\n"
        prompt += f"- 出发日期：{problem.travel_date}\n"
        prompt += f"- 总预算：{problem.budget}元\n"

        if problem.preference and problem.preference != "无特殊要求":
            prompt += f"- 交通偏好：{problem.preference}\n"

        if problem.interests:
            prompt += f"\n兴趣偏好：{', '.join(problem.interests)}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我完成完整的旅行规划：\n"
        prompt += "1. 往返交通方案（至少对比2种方案，列出航班号/车次、时间、价格）\n"
        prompt += "2. 每日详细行程安排（每天至少2-3个景点，含门票和交通）\n"
        prompt += "3. 每天的餐饮推荐（具体餐厅名称和人均价格）和住宿建议\n"
        prompt += "4. 完整预算明细（交通、住宿、餐饮、门票分项合计）"

        return prompt

    def _single_poi_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for single POI deep exploration."""
        group_desc = self._get_group_description(problem)
        poi_name = problem.interests[0] if problem.interests else "景点"

        prompt = f"我想在{problem.destination_city}深度游览{poi_name}及其周边区域。\n\n"
        prompt += f"基本信息：\n"
        prompt += f"- 出行人员：{group_desc}\n"
        prompt += f"- 日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 预算：{problem.budget}元\n"

        if len(problem.interests) > 1:
            prompt += f"\n重点关注：{', '.join(problem.interests[1:])}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += f"1. {poi_name}的最佳游览路线（上午/下午分段安排，标注每段时间）\n"
        prompt += "2. 周边至少推荐4-5个值得一去的景点、餐厅或特色店铺\n"
        prompt += "3. 各景点间的交通方式、具体距离和步行/乘车时间\n"
        prompt += "4. 每个景点的门票价格、开放时间和注意事项\n"
        prompt += "5. 总花费预估（门票+餐饮+交通分项明细）"

        return prompt

    def _food_tour_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for food tour."""
        group_desc = self._get_group_description(problem)
        food_theme = problem.interests[1] if len(problem.interests) > 1 else "当地美食"

        prompt = f"我想在{problem.destination_city}来一次{food_theme}之旅"
        if problem.num_days > 1:
            prompt += f"，计划{problem.num_days}天"
        prompt += "。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出行人员：{group_desc}\n"
        prompt += f"- 日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 餐饮预算：{problem.budget}元\n"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += "1. 至少推荐6-8家必吃的特色餐厅/店铺（含具体店名和招牌菜）\n"
        prompt += "2. 按区域或时间段规划美食路线（标注每家店的区域位置）\n"
        prompt += "3. 各店铺之间的交通方式、步行距离和所需时间\n"
        prompt += "4. 每家店的人均消费和总预算分配\n"
        prompt += "5. 用餐时间建议和排队预估"

        return prompt

    def _business_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for business trip."""
        purpose = problem.interests[0] if problem.interests else "商务出行"

        prompt = f"我因{purpose}需要从{problem.origin_city}前往{problem.destination_city}"
        if problem.num_days > 1:
            prompt += f"，预计{problem.num_days}天"
        prompt += "。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出发日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 差旅预算：{problem.budget}元\n"
        if problem.preference:
            prompt += f"- 交通偏好：{problem.preference}\n"
        if problem.arrival_constraint and problem.arrival_constraint != "无时间限制":
            prompt += f"- 到达要求：{problem.arrival_constraint}\n"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += "1. 至少对比2-3种交通方案（列出航班号/车次、时间、价格）\n"
        prompt += "2. 推荐2-3家商务区附近的酒店（含价格、距离会场/客户距离）\n"
        prompt += "3. 每天的餐饮安排（具体餐厅名称和商务宴请选择）\n"
        prompt += "4. 详细差旅费用预估（交通+住宿+餐饮分项明细）"

        return prompt

    def _family_study_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for family educational trip."""
        study_theme = problem.interests[0] if problem.interests else "亲子研学"

        prompt = f"我们一家{problem.group_size}口想去{problem.destination_city}进行一次{study_theme}之旅"
        prompt += f"，计划{problem.num_days}天。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出发日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 总预算：{problem.budget}元\n"

        if problem.interests:
            prompt += f"\n学习主题：{', '.join(problem.interests)}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += "1. 每天至少安排1个教育活动和1个休闲活动（标注适合年龄段）\n"
        prompt += "2. 每日详细行程安排（注意儿童体力，上午/下午分段，含各景点间交通）\n"
        prompt += "3. 每天推荐亲子友好的餐厅（具体名称和儿童菜品）和住宿（家庭房价格）\n"
        prompt += "4. 每个景点的门票（儿童票/家庭票价格）、开放时间和教育亮点\n"
        prompt += "5. 总预算明细（门票+餐饮+住宿+交通分项）"

        return prompt

    def _get_group_description(self, problem: TravelProblem) -> str:
        """Get group description in Chinese."""
        desc_map = {
            "solo": "独自一人",
            "couple": "情侣二人",
            "family": f"{problem.group_size}口之家",
            "friends": f"{problem.group_size}位朋友结伴",
            "elderly": f"{problem.group_size}位老年人",
        }
        return desc_map.get(problem.group_type, f"{problem.group_size}人")


# Singleton
_default_generator = None


def get_generator() -> ProblemGenerator:
    """Get default generator instance."""
    global _default_generator
    if _default_generator is None:
        _default_generator = ProblemGenerator()
    return _default_generator

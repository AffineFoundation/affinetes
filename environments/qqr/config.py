"""
QQR Travel Planning Evaluation Environment - Configuration

Tools: AMap (POI search, routing, weather) and Transport (flight/train search)
"""

import os
from typing import List

# API Keys
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")
PYTHONPATH = os.getenv("PYTHONPATH", "")

# Required tools by problem type
REQUIRED_TOOLS_BY_TYPE = {
    "intercity": ["poi_search", "direction", "weather", "search_flights", "search_train_tickets"],
    "multiday": ["poi_search", "around_search", "direction", "weather"],
    "hybrid": ["poi_search", "around_search", "direction", "weather", "search_flights", "search_train_tickets"],
    "single_poi": ["poi_search", "around_search", "direction", "weather"],
    "food_tour": ["poi_search", "around_search", "direction", "weather"],
    "business": ["poi_search", "direction", "weather", "search_flights", "search_train_tickets"],
    "family_study": ["poi_search", "around_search", "direction", "weather"],
}

# Cities for problem generation (80+ cities)
MAJOR_CITIES: List[str] = [
    # Tier 1 cities
    "北京", "上海", "广州", "深圳",
    # New Tier 1 cities
    "杭州", "成都", "西安", "重庆", "南京", "武汉", "长沙", "苏州", "天津",
    # Tier 2 cities
    "青岛", "厦门", "大连", "郑州", "济南", "沈阳", "哈尔滨", "福州",
    "合肥", "南昌", "贵阳", "南宁", "海口", "太原", "兰州",
    "石家庄", "长春", "昆明", "宁波", "无锡", "温州", "烟台",
    # Tourist cities
    "三亚", "桂林", "丽江", "张家界", "黄山", "九寨沟",
    "洛阳", "泉州", "大理", "威海", "珠海", "北海",
    "秦皇岛", "敦煌", "景德镇", "婺源", "平遥",
    # Sichuan tourist corridor
    "乐山", "都江堰", "峨眉山", "稻城",
    # Western cities
    "银川", "西宁", "呼和浩特", "乌鲁木齐", "拉萨",
    # Jiangsu/Zhejiang extended
    "常州", "徐州", "连云港", "扬州",
    # Yunnan extended
    "腾冲",
    # Guangxi extended
    "阳朔",
    # Hunan extended
    "凤凰古城",
    # Gansu extended
    "张掖", "嘉峪关",
    # Hebei extended
    "承德",
]

# City pairs by distance
CITY_PAIRS = {
    "short": [
        ("北京", "天津"), ("上海", "杭州"), ("广州", "深圳"),
        ("成都", "重庆"), ("南京", "苏州"), ("西安", "郑州"),
        ("上海", "苏州"), ("南京", "合肥"), ("广州", "珠海"),
        ("杭州", "宁波"), ("武汉", "长沙"), ("济南", "青岛"),
        ("福州", "厦门"), ("昆明", "大理"), ("成都", "乐山"),
        ("长沙", "张家界"), ("桂林", "阳朔"), ("沈阳", "大连"),
        ("无锡", "常州"), ("成都", "都江堰"), ("西安", "洛阳"),
    ],
    "medium": [
        ("北京", "上海"), ("上海", "南京"), ("广州", "长沙"),
        ("杭州", "厦门"), ("成都", "西安"), ("北京", "青岛"),
        ("北京", "济南"), ("上海", "武汉"), ("广州", "福州"),
        ("北京", "郑州"), ("上海", "合肥"), ("深圳", "厦门"),
        ("成都", "贵阳"), ("武汉", "南京"), ("北京", "大连"),
        ("上海", "长沙"), ("广州", "南宁"), ("昆明", "贵阳"),
        ("杭州", "南京"), ("北京", "沈阳"), ("哈尔滨", "长春"),
        ("西安", "兰州"), ("南京", "武汉"),
    ],
    "long": [
        ("北京", "广州"), ("上海", "成都"), ("北京", "昆明"),
        ("上海", "西安"), ("深圳", "北京"), ("广州", "杭州"),
        ("北京", "三亚"), ("上海", "三亚"), ("北京", "成都"),
        ("广州", "西安"), ("上海", "昆明"), ("深圳", "成都"),
        ("北京", "厦门"), ("上海", "海口"), ("广州", "北京"),
        ("北京", "哈尔滨"), ("上海", "大连"), ("广州", "贵阳"),
        ("成都", "哈尔滨"), ("北京", "乌鲁木齐"), ("上海", "丽江"),
        ("深圳", "西安"), ("北京", "南宁"), ("广州", "济南"),
    ],
}

# Problem types (7 types for expanded task space)
PROBLEM_TYPES = [
    "intercity", "multiday", "hybrid",
    "single_poi", "food_tour", "business", "family_study",
]

# Tool coverage threshold by problem type
# Non-transport types use 0.5 (half of required tools must be called)
# because direction/around_search require coordinates that models often
# cannot extract from text-format tool results (Chutes API workaround)
REQUIRED_TOOLS_THRESHOLD = {
    "intercity": 0.6,
    "multiday": 0.5,
    "hybrid": 0.5,
    "single_poi": 0.5,
    "food_tour": 0.5,
    "business": 0.6,
    "family_study": 0.5,
}

# Core tools that must all be called
# Only poi_search is truly irreplaceable — direction/around_search require
# coordinates and are quality-enhancing but not strictly necessary
CORE_TOOLS_BY_TYPE = {
    "intercity": set(),  # No core tools; REQUIRES_TRANSPORT + 60% threshold sufficient
    "multiday": {"poi_search"},
    "hybrid": {"poi_search"},
    "single_poi": {"poi_search"},
    "food_tour": {"poi_search"},
    "business": {"poi_search"},
    "family_study": {"poi_search"},
}

# Transport tools - at least one must be called for transport-related problems
TRANSPORT_TOOLS = {"search_flights", "search_train_tickets"}

# Problem types that require transport information
REQUIRES_TRANSPORT = {"intercity", "hybrid", "business"}

# Conflicting constraint pairs (used by DifficultyProfile for constraint injection)
CONFLICTING_CONSTRAINT_PAIRS = [
    ("深度体验", "轻松休闲"),
    ("预算优先", "舒适优先"),
    ("速度优先", "经济优先"),
    ("避开人群", "热门景点打卡"),
    ("公共交通优先", "速度优先"),
    ("素食要求", "美食探店"),
    ("不乘坐红眼航班", "经济优先"),
]

# Anti-memorization: require minimum info consistency ratio
INFO_CONSISTENCY_MIN_RATIO = 0.4  # Must use at least 40% of available tool info

# IC minimum quantity gate: when tool returns many facts per category,
# require a proportional number of matches (not just 1)
IC_MIN_QUANTITY_THRESHOLD = 4   # Only apply when tool has >= 4 facts in category
IC_MIN_QUANTITY_RATIO = 0.3     # Require matching at least 30% of tool facts
IC_MIN_QUANTITY_CAP = 3         # Never require more than 3 matches per category
IC_BELOW_MINIMUM_SCALE = 0.5   # Cap category score at 50% when below minimum

# Format validation thresholds
FORMAT_MIN_LENGTH = 200         # Minimum output length (chars) for valid format

# Anti-echo: Tier-2 proximity and structural credit
TIER2_PROXIMITY_DISTANCE = 500    # Max chars between keyword and fact for Tier-2 half-score
STRUCTURAL_CREDIT_RATIO = 0.1    # Structural credit without tool data (was 0.2)

# IC context-sensitive matching
IC_OUT_OF_CONTEXT_WEIGHT = 0.5    # Weight for facts not near relevant context keywords

# Code-determined tool_info_used thresholds
# IC/Comp are based on epoch-salted fact overlap — not forgeable
# Production data: genuine tool use → IC≈25, Comp≈29; fabricated → IC≈0, Comp≈0
CODE_TOOL_USED_IC_THRESHOLD = 6.0           # Transport types (intercity, hybrid, business)
CODE_TOOL_USED_COMP_THRESHOLD = 6.0         # Transport types
CODE_TOOL_USED_IC_THRESHOLD_NONTRANSPORT = 4.0   # Non-transport (fewer verifiable categories)
CODE_TOOL_USED_COMP_THRESHOLD_NONTRANSPORT = 4.0

# Anti-memorization: require specific POI names from tools
ENABLE_POI_VERIFICATION = True
MIN_POI_MATCH_COUNT = 2  # Must match at least 2 POI names from tool results

# ============================================================================
# TRANSPORT GROUNDING: Ensure transport info comes from tools
# This is the key anti-memorization measure for intercity/hybrid problems
# ============================================================================
ENABLE_TRANSPORT_GROUNDING = True  # Master switch for transport grounding

# Transport grounding configuration
# Only verifies transport-related claims, not general content
TRANSPORT_GROUNDING_CONFIG = {
    # Flight/train IDs: 100% must come from tools (no fabrication allowed)
    "transport_id_match_ratio": 1.0,

    # Transport prices: must match within tolerance
    "transport_price_tolerance": 0.15,  # 15% tolerance
    "transport_price_match_ratio": 0.7,  # 70% of transport prices must match

    # Transport times: must match tool results
    "transport_time_match_ratio": 0.7,  # 70% of transport times must match

    # Maximum fabrication ratio for transport claims specifically
    "max_transport_fabrication_ratio": 0.2,  # Max 20% unverified transport claims
}

# Code score weights (max 50) - tool_coverage/validity are gating prerequisites (0 points)
# All scoring weight on info_consistency + completeness (the hard, grounding-based dimensions)
CODE_SCORE_WEIGHTS = {
    "tool_coverage": 0.0,
    "tool_validity": 0.0,
    "info_consistency": 25.0,
    "completeness": 25.0,
}

# Info consistency threshold: ratio/divisor normalization per category
# 0.5 means 50%+ overlap per category = full score (was 0.3)
INFO_CONSISTENCY_RATIO_DIVISOR = 0.6

# Minimum category breadth: if fewer than this many categories matched
# AND total categories >= MIN_BREADTH_TOTAL, apply breadth penalty
INFO_CONSISTENCY_MIN_BREADTH_TOTAL = 3
IC_BREADTH_PENALTY_MULTIPLIER = 0.3  # Harsh penalty for narrow category coverage

# Fabrication penalty (deducted from code score, 25% of code max)
FABRICATION_PENALTY_MAX = -12.5

# Hard constraint penalty multipliers for soft constraints
# 0.0 = hard fail (total score = 0), 0.5 = 50% penalty, 1.0 = no penalty
HARD_CONSTRAINT_PENALTIES = {
    "format_valid": 0.15,           # Near-fail — output is garbage/off-topic but gives RL gradient
    "tool_info_used": 0.0,          # Hard fail - anti-hack
    "required_tools_called": 0.5,   # Soft - missing tools gets 50% penalty
    "poi_names_verified": 0.7,      # Soft - POI mismatch gets 30% penalty
    "transport_grounded": 0.3,      # Soft - transport fabrication gets 70% penalty (was 0.5)
    "tool_quality": 0.5,            # Soft - low tool_coverage OR tool_validity gets 50% penalty
}

# LLM-code smooth coupling factor
# llm_score *= min(1.0, code_total / (TOTAL_CODE_SCORE * factor))
LLM_CODE_RATIO_FACTOR = 0.75

# LLM score weights (max 50) - elevated to 50/50 split for anti-hack
# "analysis_depth" replaces "informativeness" to penalize data-dumping
LLM_SCORE_WEIGHTS = {
    "practicality": 12.5,
    "logic": 12.5,
    "user_experience": 12.5,
    "analysis_depth": 12.5,
}

# Total scores
TOTAL_CODE_SCORE = sum(CODE_SCORE_WEIGHTS.values())
TOTAL_LLM_SCORE = sum(LLM_SCORE_WEIGHTS.values())

# Minimum per-person one-way transport cost estimate by distance category.
# Based on mock_transport pricing formulas:
#   K-class train: 0.12 CNY/km, min 30 CNY
#   D-class train: 0.31 CNY/km
#   Economy flight: distance * 0.5 * 0.55 + 100
MIN_TRANSPORT_COST = {
    "short": 50,    # ~300km K-class ≈ 36 CNY, conservative 50
    "medium": 150,  # ~800km K-class ≈ 96 CNY, D-class ≈ 137 CNY, conservative 150
    "long": 300,    # ~2000km K-class ≈ 240 CNY, cheapest flight ≈ 650 CNY, conservative 300
}

# Evaluation settings
MAX_TOOL_STEPS = 15

# System prompt for the travel planning agent
SYSTEM_PROMPT = """你是一个专业的旅行规划助手，能够帮助用户规划旅行行程。

## 可用工具

你可以使用以下工具获取真实信息：

1. **poi_search(address, region)** - 搜索地点信息
   - address: 地点名称或关键词（如"西湖"、"火车站"）
   - region: 可选，城市名称用于缩小范围

2. **around_search(location, radius, keyword, region)** - 周边搜索
   - location: 中心点坐标（经度,纬度）
   - radius: 搜索半径（米，最大50000）
   - keyword: 搜索关键词
   - region: 可选，城市名称

3. **direction(origin, destination, mode, waypoints)** - 路线规划
   - origin: 起点坐标（经度,纬度）
   - destination: 终点坐标（经度,纬度）
   - mode: 出行方式（driving/walking/bicycling/transit）
   - waypoints: 可选，途经点列表

4. **weather(city)** - 天气查询
   - city: 城市名称

5. **search_flights(date, from_city, to_city)** - 航班搜索
   - date: 日期（YYYY-MM-DD格式）
   - from_city: 出发城市
   - to_city: 到达城市

6. **search_train_tickets(date, from_city, to_city, ...)** - 火车票搜索
   - date: 日期（YYYY-MM-DD格式）
   - from_city: 出发城市
   - to_city: 到达城市
   - 其他参数：城市代码和坐标（可从poi_search获取）

## 工作流程

1. **第一步**：调用 poi_search 搜索景点、酒店、餐厅等地点信息
2. **第二步**：调用 weather 查询目的地天气预报（**必须调用**）
3. **第三步**：调用 direction 规划景点之间的路线（**必须调用**）
4. **第四步**：如需要，调用 around_search 搜索周边设施
5. **最后**：根据所有工具返回的信息，提供详细的规划方案

## 重要要求

- **必须**调用多种工具获取完整信息，不能只使用 poi_search
- **必须**调用 weather 工具查询天气，这对旅行规划至关重要
- **必须**调用 direction 工具规划路线，提供具体的交通时间和距离
- 最终方案中的信息必须与工具返回的结果一致
- 不要编造工具没有返回的信息
- 在获取足够信息之前，不要急于给出最终规划
"""

# OpenAI Function Calling Schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "poi_search",
            "description": "搜索地点信息（景点、酒店、餐厅等POI）。返回地点的名称、地址、坐标等信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "搜索关键词或地址，如'西湖'、'北京火车站'、'餐厅'"
                    },
                    "region": {
                        "type": "string",
                        "description": "城市名称，用于缩小搜索范围，如'杭州'、'北京'"
                    }
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "around_search",
            "description": "周边搜索。在指定中心点和半径范围内搜索地点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "中心点坐标，格式为'经度,纬度'，如'120.15,30.28'"
                    },
                    "radius": {
                        "type": "integer",
                        "description": "搜索半径（米），范围0-50000"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，如'餐厅'、'酒店'"
                    },
                    "region": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "direction",
            "description": "路线规划。计算两点之间的路线、距离和时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "起点坐标，格式为'经度,纬度'"
                    },
                    "destination": {
                        "type": "string",
                        "description": "终点坐标，格式为'经度,纬度'"
                    },
                    "mode": {
                        "type": "string",
                        "description": "出行方式：driving（驾车）、walking（步行）、bicycling（骑行）、transit（公交）",
                        "enum": ["driving", "walking", "bicycling", "transit"]
                    },
                    "waypoints": {
                        "type": "string",
                        "description": "途经点坐标列表，用分号分隔"
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "天气查询。获取指定城市的天气预报。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'杭州'、'北京'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "航班搜索。查询两个城市之间的航班信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "出发日期，格式YYYY-MM-DD"
                    },
                    "from_city": {
                        "type": "string",
                        "description": "出发城市中文名"
                    },
                    "to_city": {
                        "type": "string",
                        "description": "到达城市中文名"
                    }
                },
                "required": ["date", "from_city", "to_city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_train_tickets",
            "description": "火车票搜索。查询两个城市之间的火车票信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "出发日期，格式YYYY-MM-DD"
                    },
                    "from_city": {
                        "type": "string",
                        "description": "出发城市中文名"
                    },
                    "to_city": {
                        "type": "string",
                        "description": "到达城市中文名"
                    },
                    "from_city_adcode": {
                        "type": "string",
                        "description": "出发城市行政区划代码"
                    },
                    "to_city_adcode": {
                        "type": "string",
                        "description": "到达城市行政区划代码"
                    },
                    "from_lat": {
                        "type": "string",
                        "description": "出发城市纬度"
                    },
                    "from_lon": {
                        "type": "string",
                        "description": "出发城市经度"
                    },
                    "to_lat": {
                        "type": "string",
                        "description": "到达城市纬度"
                    },
                    "to_lon": {
                        "type": "string",
                        "description": "到达城市经度"
                    }
                },
                "required": ["date", "from_city", "to_city"]
            }
        }
    }
]

# ============================================================================
# CITY_PAIRS SAFETY ASSERTIONS
# Prevent future maintainers from adding city pairs with no transport options.
# ============================================================================
CITIES_WITHOUT_AIRPORTS = {
    "苏州", "大理", "乐山", "峨眉山", "都江堰", "平遥",
    "承德", "秦皇岛", "凤凰古城", "婺源", "阳朔",
}
CITIES_WITHOUT_TRAINS = {
    "九寨沟", "稻城", "腾冲", "凤凰古城",
}

# Verify: every city in CITY_PAIRS has at least one transport mode
for _dist, _pairs in CITY_PAIRS.items():
    for _o, _d in _pairs:
        assert (_o not in CITIES_WITHOUT_AIRPORTS or _o not in CITIES_WITHOUT_TRAINS), \
            f"CITY_PAIR ({_o}, {_d}): {_o} has no transport"
        assert (_d not in CITIES_WITHOUT_AIRPORTS or _d not in CITIES_WITHOUT_TRAINS), \
            f"CITY_PAIR ({_o}, {_d}): {_d} has no transport"

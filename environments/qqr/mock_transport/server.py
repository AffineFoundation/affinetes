"""
Deterministic mock transport server.

Generates flight and train data algorithmically from a seed derived from
(date, from_city, to_city). Same inputs always produce the same outputs.

No LLM dependency - pure algorithmic generation.
"""
import hashlib
import json
import logging
import os
import random
import time
from typing import List

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("MockTransport", log_level="WARNING")

# Anti-memorization: epoch salt changes weekly so transport data can't be memorized
# Set via TRANSPORT_SALT env var, or auto-generated from current week
TRANSPORT_SALT = os.getenv("TRANSPORT_SALT", "")


# ============================================================================
# City Data
# ============================================================================

# Major airports per city
CITY_AIRPORTS = {
    "北京": ["首都国际机场", "大兴国际机场"],
    "上海": ["浦东国际机场", "虹桥国际机场"],
    "广州": ["白云国际机场"],
    "深圳": ["宝安国际机场"],
    "杭州": ["萧山国际机场"],
    "成都": ["天府国际机场", "双流国际机场"],
    "西安": ["咸阳国际机场"],
    "重庆": ["江北国际机场"],
    "南京": ["禄口国际机场"],
    "武汉": ["天河国际机场"],
    "长沙": ["黄花国际机场"],
    "青岛": ["胶东国际机场"],
    "苏州": [],  # No major airport, uses Shanghai
    "厦门": ["高崎国际机场"],
    "大连": ["周水子国际机场"],
    "天津": ["滨海国际机场"],
    "三亚": ["凤凰国际机场"],
    "昆明": ["长水国际机场"],
    "桂林": ["两江国际机场"],
    "丽江": ["三义国际机场"],
    "张家界": ["荷花国际机场"],
    "黄山": ["屯溪国际机场"],
    "九寨沟": ["黄龙九寨沟机场"],
    "郑州": ["新郑国际机场"],
    # Expanded cities (Phase 2)
    "洛阳": ["北郊机场"],
    "泉州": ["晋江国际机场"],
    "大理": [],
    "威海": ["大水泊国际机场"],
    "哈尔滨": ["太平国际机场"],
    "沈阳": ["桃仙国际机场"],
    "济南": ["遥墙国际机场"],
    "福州": ["长乐国际机场"],
    "合肥": ["新桥国际机场"],
    "南昌": ["昌北国际机场"],
    "贵阳": ["龙洞堡国际机场"],
    "南宁": ["吴圩国际机场"],
    "海口": ["美兰国际机场"],
    "太原": ["武宿国际机场"],
    "兰州": ["中川国际机场"],
    "银川": ["河东国际机场"],
    "西宁": ["曹家堡国际机场"],
    "呼和浩特": ["白塔国际机场"],
    "乌鲁木齐": ["地窝堡国际机场"],
    "拉萨": ["贡嘎机场"],
    "珠海": ["金湾机场"],
    "无锡": ["硕放机场"],
    "温州": ["龙湾国际机场"],
    "宁波": ["栎社国际机场"],
    "烟台": ["蓬莱国际机场"],
    "石家庄": ["正定国际机场"],
    "长春": ["龙嘉国际机场"],
    "常州": ["奔牛国际机场"],
    "徐州": ["观音国际机场"],
    "连云港": ["花果山国际机场"],
    "扬州": ["泰州扬州机场"],
    "秦皇岛": [],
    "承德": [],
    "敦煌": ["莫高国际机场"],
    "张掖": ["甘州机场"],
    "嘉峪关": ["机场"],
    "腾冲": ["驼峰机场"],
    "景德镇": ["罗家机场"],
    "北海": ["福成机场"],
    "阳朔": [],
    "凤凰古城": [],
    "婺源": [],
    "平遥": [],
    "乐山": [],
    "都江堰": [],
    "峨眉山": [],
    "稻城": ["亚丁机场"],
}

# Train stations per city
CITY_TRAIN_STATIONS = {
    "北京": ["北京南站", "北京西站", "北京站", "北京北站"],
    "上海": ["上海虹桥站", "上海站", "上海南站"],
    "广州": ["广州南站", "广州站", "广州东站"],
    "深圳": ["深圳北站", "深圳站", "福田站"],
    "杭州": ["杭州东站", "杭州站", "杭州西站"],
    "成都": ["成都东站", "成都站", "成都南站"],
    "西安": ["西安北站", "西安站"],
    "重庆": ["重庆北站", "重庆西站", "重庆站"],
    "南京": ["南京南站", "南京站"],
    "武汉": ["武汉站", "汉口站", "武昌站"],
    "长沙": ["长沙南站", "长沙站"],
    "青岛": ["青岛站", "青岛北站"],
    "苏州": ["苏州站", "苏州北站"],
    "厦门": ["厦门站", "厦门北站"],
    "大连": ["大连站", "大连北站"],
    "天津": ["天津站", "天津南站", "天津西站"],
    "三亚": ["三亚站"],
    "昆明": ["昆明南站", "昆明站"],
    "桂林": ["桂林站", "桂林北站"],
    "丽江": ["丽江站"],
    "张家界": ["张家界西站"],
    "黄山": ["黄山北站"],
    "九寨沟": [],
    "郑州": ["郑州东站", "郑州站"],
    # Expanded cities
    "洛阳": ["洛阳龙门站", "洛阳站"],
    "泉州": ["泉州站"],
    "大理": ["大理站"],
    "威海": ["威海站"],
    "哈尔滨": ["哈尔滨西站", "哈尔滨站"],
    "沈阳": ["沈阳北站", "沈阳站"],
    "济南": ["济南西站", "济南站"],
    "福州": ["福州站", "福州南站"],
    "合肥": ["合肥南站", "合肥站"],
    "南昌": ["南昌西站", "南昌站"],
    "贵阳": ["贵阳北站", "贵阳站"],
    "南宁": ["南宁东站", "南宁站"],
    "海口": ["海口东站", "海口站"],
    "太原": ["太原南站", "太原站"],
    "兰州": ["兰州西站", "兰州站"],
    "银川": ["银川站"],
    "西宁": ["西宁站"],
    "呼和浩特": ["呼和浩特东站"],
    "乌鲁木齐": ["乌鲁木齐站"],
    "拉萨": ["拉萨站"],
    "珠海": ["珠海站"],
    "无锡": ["无锡站", "无锡东站"],
    "温州": ["温州南站"],
    "宁波": ["宁波站"],
    "烟台": ["烟台站", "烟台南站"],
    "石家庄": ["石家庄站"],
    "长春": ["长春站", "长春西站"],
    "常州": ["常州站", "常州北站"],
    "徐州": ["徐州东站", "徐州站"],
    "连云港": ["连云港站"],
    "扬州": ["扬州东站"],
    "秦皇岛": ["秦皇岛站"],
    "承德": ["承德南站"],
    "敦煌": ["敦煌站"],
    "张掖": ["张掖西站"],
    "嘉峪关": ["嘉峪关南站"],
    "腾冲": [],
    "景德镇": ["景德镇北站"],
    "北海": ["北海站"],
    "阳朔": ["阳朔站"],
    "凤凰古城": [],
    "婺源": ["婺源站"],
    "平遥": ["平遥古城站"],
    "乐山": ["乐山站"],
    "都江堰": ["都江堰站"],
    "峨眉山": ["峨眉山站"],
    "稻城": [],
}

# Approximate distances between major city pairs (km, for pricing)
# If pair not found, estimate from a default based on region
CITY_DISTANCES = {
    ("北京", "上海"): 1200, ("北京", "广州"): 2100, ("北京", "深圳"): 2200,
    ("北京", "杭州"): 1300, ("北京", "成都"): 1800, ("北京", "西安"): 1100,
    ("北京", "重庆"): 1700, ("北京", "南京"): 1000, ("北京", "武汉"): 1200,
    ("北京", "长沙"): 1500, ("北京", "青岛"): 700, ("北京", "厦门"): 1800,
    ("北京", "大连"): 900, ("北京", "天津"): 120, ("北京", "三亚"): 2800,
    ("北京", "昆明"): 2500, ("北京", "桂林"): 1900, ("北京", "丽江"): 2600,
    ("北京", "郑州"): 700, ("北京", "哈尔滨"): 1200, ("北京", "沈阳"): 700,
    ("北京", "济南"): 400, ("北京", "福州"): 1700, ("北京", "合肥"): 1000,
    ("北京", "南昌"): 1400, ("北京", "贵阳"): 2100, ("北京", "南宁"): 2300,
    ("北京", "海口"): 2700, ("北京", "太原"): 500, ("北京", "兰州"): 1500,
    ("北京", "乌鲁木齐"): 3000, ("北京", "拉萨"): 3700,
    ("上海", "广州"): 1500, ("上海", "深圳"): 1500, ("上海", "杭州"): 180,
    ("上海", "成都"): 2000, ("上海", "西安"): 1500, ("上海", "重庆"): 1800,
    ("上海", "南京"): 300, ("上海", "武汉"): 800, ("上海", "长沙"): 1000,
    ("上海", "青岛"): 800, ("上海", "苏州"): 100, ("上海", "厦门"): 800,
    ("上海", "大连"): 1200, ("上海", "天津"): 1100, ("上海", "三亚"): 2000,
    ("上海", "昆明"): 2200, ("上海", "桂林"): 1500, ("上海", "丽江"): 2300,
    ("上海", "福州"): 700, ("上海", "合肥"): 450, ("上海", "南昌"): 700,
    ("上海", "贵阳"): 1700, ("上海", "海口"): 1900,
    ("广州", "深圳"): 140, ("广州", "杭州"): 1300, ("广州", "成都"): 1600,
    ("广州", "西安"): 1600, ("广州", "重庆"): 1300, ("广州", "南京"): 1300,
    ("广州", "武汉"): 1000, ("广州", "长沙"): 700, ("广州", "厦门"): 600,
    ("广州", "三亚"): 800, ("广州", "昆明"): 1500, ("广州", "桂林"): 500,
    ("广州", "南宁"): 600, ("广州", "海口"): 600, ("广州", "贵阳"): 900,
    ("广州", "福州"): 800, ("广州", "南昌"): 800,
    ("深圳", "杭州"): 1300, ("深圳", "成都"): 1700, ("深圳", "厦门"): 500,
    ("深圳", "北京"): 2200,
    ("成都", "重庆"): 300, ("成都", "西安"): 700, ("成都", "昆明"): 800,
    ("成都", "贵阳"): 700, ("成都", "拉萨"): 1600, ("成都", "兰州"): 700,
    ("杭州", "厦门"): 700, ("杭州", "南京"): 300, ("杭州", "武汉"): 800,
    ("南京", "苏州"): 200, ("南京", "武汉"): 500, ("南京", "合肥"): 300,
    ("武汉", "长沙"): 350,
    ("西安", "郑州"): 500, ("西安", "兰州"): 600, ("西安", "成都"): 700,
    ("昆明", "大理"): 350, ("昆明", "丽江"): 500, ("昆明", "贵阳"): 500,
}

# Airline codes pool
AIRLINE_CODES = [
    "CA", "MU", "CZ", "HU", "3U", "HO", "ZH", "MF",
    "FM", "GS", "SC", "KN", "JD", "EU", "TV",
]

# Train type config: (prefix, speed_kmh, base_price_per_km)
TRAIN_TYPES = [
    ("G", 300, 0.46),   # High-speed
    ("D", 250, 0.31),   # EMU
    ("C", 300, 0.46),   # Intercity high-speed
    ("Z", 120, 0.16),   # Direct express
    ("T", 100, 0.14),   # Express
    ("K", 80, 0.12),    # Fast
]


# ============================================================================
# Deterministic Generation Logic
# ============================================================================

def _make_seed(date: str, from_city: str, to_city: str) -> int:
    """Create deterministic seed from input parameters.

    Incorporates TRANSPORT_SALT for anti-memorization: same inputs produce
    different data across evaluation epochs (weekly rotation).
    """
    salt = TRANSPORT_SALT
    if not salt:
        # Auto-generate weekly salt if not set via env var
        salt = str(int(time.time()) // (7 * 86400))
    key = f"{salt}|{date}|{from_city}|{to_city}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _get_distance(from_city: str, to_city: str) -> int:
    """Get approximate distance between two cities in km."""
    pair = (from_city, to_city)
    if pair in CITY_DISTANCES:
        return CITY_DISTANCES[pair]
    reverse = (to_city, from_city)
    if reverse in CITY_DISTANCES:
        return CITY_DISTANCES[reverse]
    # Default estimate: salt-independent, symmetric hash for unlisted pairs
    cities_sorted = tuple(sorted([from_city, to_city]))
    key = f"distance|{cities_sorted[0]}|{cities_sorted[1]}"
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return 500 + (seed % 2000)


def _generate_flights(
    date: str, from_city: str, to_city: str
) -> List[str]:
    """Generate deterministic flight data."""
    seed = _make_seed(date, from_city, to_city)
    rng = random.Random(seed)

    distance = _get_distance(from_city, to_city)

    # Cities without airports get no flights
    from_airports = CITY_AIRPORTS.get(from_city, [])
    to_airports = CITY_AIRPORTS.get(to_city, [])
    if not from_airports or not to_airports:
        return []

    # Number of flights: 8-15 based on distance and city size
    num_flights = rng.randint(8, 15)

    # Generate departure times: normal daytime + red-eye flights
    num_redeye = rng.randint(1, 2)
    num_normal = num_flights - num_redeye
    normal_deps = sorted(rng.sample(range(300, 1380), min(num_normal, 1080)))
    redeye_deps = [rng.randint(1380, 1500) for _ in range(num_redeye)]  # 23:00-01:00
    dep_minutes = sorted(normal_deps + redeye_deps)

    flights = []
    used_flight_numbers = set()

    for i, dep_min in enumerate(dep_minutes):
        # Airline selection
        airline = rng.choice(AIRLINE_CODES)

        # Flight number: deterministic from seed + index
        flight_id = None
        for _ in range(10):
            flight_num = rng.randint(100, 9999)
            candidate = f"{airline}{flight_num}"
            if candidate not in used_flight_numbers:
                flight_id = candidate
                used_flight_numbers.add(flight_id)
                break
        if flight_id is None:
            continue  # Skip this flight if no unique ID found

        # Airports
        dep_airport = rng.choice(from_airports)
        arr_airport = rng.choice(to_airports)

        # Flight duration based on distance (avg 800km/h with overhead)
        flight_hours = distance / 750
        flight_hours *= (0.9 + rng.random() * 0.2)  # +/- 10% variation
        flight_minutes = int(flight_hours * 60)
        flight_minutes = max(60, flight_minutes)  # At least 1 hour

        # Arrival time
        arr_min = dep_min + flight_minutes
        dep_h, dep_m = divmod(dep_min, 60)
        arr_h, arr_m = divmod(arr_min, 60)

        # Handle next-day arrival for red-eye flights
        next_day = ""
        if arr_h >= 24:
            arr_h -= 24
            next_day = "(次日)"

        # Normalize departure hour for display (handle >24h)
        display_dep_h = dep_h % 24 if dep_h >= 24 else dep_h

        # Price based on distance + skewed variation
        base_price = distance * 0.5 + 100
        is_redeye = dep_min >= 1380 or dep_min < 300
        # Price skew: 30% economy + 70% premium
        if rng.random() < 0.3:
            price_variation = rng.uniform(0.55, 0.85)  # Economy option
        else:
            price_variation = rng.uniform(1.1, 1.7)    # Premium option
        price = round(base_price * price_variation)
        # Red-eye discount
        if is_redeye:
            price = int(price * rng.uniform(0.6, 0.8))
        price = max(200, min(8000, price))

        # Duration string
        dur_h = flight_minutes // 60
        dur_m = flight_minutes % 60
        duration_str = f"{dur_h}小时{dur_m}分" if dur_m > 0 else f"{dur_h}小时"

        record = (
            f"航班 {flight_id}，价格{price}元，"
            f"{display_dep_h:02d}:{dep_m:02d}从{dep_airport}出发，"
            f"{arr_h:02d}:{arr_m:02d}{next_day}到达{arr_airport}，"
            f"飞行时长{duration_str}"
        )
        flights.append(record)

    return flights


def _generate_trains(
    date: str, from_city: str, to_city: str
) -> List[str]:
    """Generate deterministic train data."""
    seed = _make_seed(date, from_city, to_city)
    rng = random.Random(seed)

    distance = _get_distance(from_city, to_city)

    from_stations = CITY_TRAIN_STATIONS.get(from_city, [])
    to_stations = CITY_TRAIN_STATIONS.get(to_city, [])
    if not from_stations or not to_stations:
        return []

    # Select which train types are available based on distance
    available_types = []
    if distance <= 500:
        # Short distance: prefer G, D, C
        available_types = [("G", 300, 0.46), ("D", 250, 0.31), ("C", 300, 0.46)]
    elif distance <= 1500:
        # Medium: G, D, Z, T
        available_types = [("G", 300, 0.46), ("D", 250, 0.31), ("Z", 120, 0.16), ("T", 100, 0.14)]
    else:
        # Long: all types
        available_types = TRAIN_TYPES[:]

    num_trains = rng.randint(8, 15)
    dep_minutes = sorted(rng.sample(range(360, 1380), num_trains))  # 06:00-23:00

    trains = []
    used_train_numbers = set()

    for i, dep_min in enumerate(dep_minutes):
        # Select train type
        train_type = rng.choice(available_types)
        prefix, speed, price_per_km = train_type

        # Train number
        train_id = None
        for _ in range(10):
            if prefix in ("G", "D", "C"):
                train_num = rng.randint(1, 9999)
            else:
                train_num = rng.randint(1, 999)
            candidate = f"{prefix}{train_num}"
            if candidate not in used_train_numbers:
                train_id = candidate
                used_train_numbers.add(train_id)
                break
        if train_id is None:
            continue  # Skip this train if no unique ID found

        # Stations
        dep_station = rng.choice(from_stations)
        arr_station = rng.choice(to_stations)

        # Duration based on distance and speed
        travel_hours = distance / speed
        travel_hours *= (0.9 + rng.random() * 0.3)  # +/- variation
        travel_minutes = int(travel_hours * 60)
        travel_minutes = max(30, travel_minutes)

        # Arrival time
        arr_min = dep_min + travel_minutes
        dep_h, dep_m = divmod(dep_min, 60)
        arr_h, arr_m = divmod(arr_min, 60)

        # Allow overnight trains for Z/T/K types
        if arr_h >= 24:
            if prefix in ("Z", "T", "K"):
                arr_h = arr_h % 24
                # Mark as next day arrival in duration
            else:
                continue

        # Price — skewed distribution for high-speed trains (G/D/C)
        base_price = distance * price_per_km
        if prefix in ("G", "D", "C"):
            # Skewed: 30% economy + 70% premium
            if rng.random() < 0.3:
                price_variation = rng.uniform(0.55, 0.85)
            else:
                price_variation = rng.uniform(1.1, 1.7)
        else:
            price_variation = rng.uniform(0.85, 1.15)  # Z/T/K keep uniform
        price = round(base_price * price_variation)
        price = max(30, min(3000, price))

        # Duration string
        dur_h = travel_minutes // 60
        dur_m = travel_minutes % 60
        if dur_h > 0:
            duration_str = f"{dur_h}小时{dur_m}分" if dur_m > 0 else f"{dur_h}小时"
        else:
            duration_str = f"{dur_m}分钟"

        record = (
            f"直达车次 {train_id}，价格{price}元，"
            f"{dep_h:02d}:{dep_m:02d}从{dep_station}出发，"
            f"{arr_h:02d}:{arr_m:02d}到达{arr_station}，"
            f"全程约{duration_str}。"
        )
        trains.append(record)

    return trains


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def search_flights(date: str, from_city: str, to_city: str) -> str:
    """
    Search flights between cities.

    date: YYYY-MM-DD format
    from_city: Departure city (Chinese name)
    to_city: Arrival city (Chinese name)
    """
    try:
        flights = _generate_flights(date, from_city, to_city)
        if not flights:
            raise ValueError(f"No flights available between {from_city} and {to_city}")
        return json.dumps(flights, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[search_flights] Error: {e}")
        raise ValueError(f"No flight information available between these cities: {e}")


@mcp.tool()
async def search_train_tickets(
    date: str,
    from_city: str,
    to_city: str,
    from_city_adcode: str = "",
    to_city_adcode: str = "",
    from_lat: str = "",
    from_lon: str = "",
    to_lat: str = "",
    to_lon: str = "",
) -> str:
    """
    Search train tickets between cities.

    date: Query date (YYYY-MM-DD format)
    from_city / to_city: Chinese city names
    from_city_adcode / to_city_adcode: Administrative codes
    from_lat, from_lon, to_lat, to_lon: Coordinates
    """
    try:
        trains = _generate_trains(date, from_city, to_city)
        if not trains:
            raise ValueError(f"No direct trains available between {from_city} and {to_city}")
        return json.dumps(trains, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[search_train_tickets] Error: {e}")
        raise ValueError(f"No direct train tickets available between these cities: {e}")


if __name__ == "__main__":
    mcp.run()

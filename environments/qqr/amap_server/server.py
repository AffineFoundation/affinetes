"""
AMap MCP Server — local fork of qqr.tools.amap with citycode caching.

Changes from upstream:
- get_citycode() cached in-memory (same coordinates always return same citycode,
  saves ~3600 reverse geocoding API calls/day from transit_direction)
- AMAP_MAPS_API_KEY read from env directly (no qqr.utils.envs dependency)
"""

import asyncio
import os

import httpx
from mcp.server.fastmcp import FastMCP

from qqr.data.markdown import json2md
from qqr.data.text import truncate_text

mcp = FastMCP("AMap", log_level="WARNING")

AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY", "")


# ============================================================================
# Reverse Geocoding with in-memory cache
# ============================================================================

_citycode_cache = {}


async def reverse_geocode(location: str):
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {"key": AMAP_MAPS_API_KEY, "location": location}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


async def get_citycode(location: str):
    # Round to 3 decimal places for cache key (~111m precision, sufficient for city lookup)
    parts = location.replace(" ", "").split(",")
    if len(parts) == 2:
        try:
            cache_key = f"{round(float(parts[0]), 3)},{round(float(parts[1]), 3)}"
        except (ValueError, OverflowError):
            cache_key = location
    else:
        cache_key = location

    if cache_key in _citycode_cache:
        return _citycode_cache[cache_key]

    result = await reverse_geocode(location)

    try:
        citycode = result["regeocode"]["addressComponent"]["citycode"]
    except Exception:
        citycode = None

    _citycode_cache[cache_key] = citycode
    return citycode


# ============================================================================
# POI Search
# ============================================================================

@mcp.tool()
async def poi_search(address: str, region: str | None = None) -> str:
    """
    Search for place information by text. Text can be a structured address
    (e.g. "北京市朝阳区望京阜荣街10号") or a POI name (e.g. "首开广场").

    Args:
        address: Text to search. Max 80 characters.
        region: City name to narrow search scope (e.g. "北京市"). Optional.
    """
    url = "https://restapi.amap.com/v5/place/text"
    params = {
        "key": AMAP_MAPS_API_KEY,
        "keywords": address,
        "show_fields": "business",
    }
    if region:
        params["region"] = region

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    pois = result.get("pois")
    if not pois:
        raise Exception("No POI data available.")

    return truncate_text(json2md(pois))


# ============================================================================
# Around Search
# ============================================================================

@mcp.tool()
async def around_search(
    location: str,
    radius: int = 5000,
    keyword: str | None = None,
    region: str | None = None,
) -> str:
    """
    Search for POIs within a circular area defined by center point and radius.

    Args:
        location: Center point coordinates ("longitude,latitude").
        radius: Search radius in meters (0-50000).
        keyword: Search keyword (e.g. "银行").
        region: City name to narrow search scope. Optional.
    """
    url = "https://restapi.amap.com/v5/place/around"
    params = {
        "key": AMAP_MAPS_API_KEY,
        "location": location,
        "radius": radius,
        "show_fields": "business",
    }
    if keyword:
        params["keywords"] = keyword
    if region:
        params["region"] = region

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    pois = result.get("pois")
    if not pois:
        raise Exception("No POI data available.")

    return truncate_text(json2md(pois))


# ============================================================================
# Direction (multiple modes)
# ============================================================================

async def driving_direction(origin: str, destination: str, waypoints: str | None = None):
    url = "https://restapi.amap.com/v5/direction/driving?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}
    if waypoints:
        params["waypoints"] = waypoints

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def walking_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/walking?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def bicycling_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/bicycling?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def electrobike_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/electrobike?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def transit_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/transit/integrated?parameters"

    citycode_origin, citycode_destination = await asyncio.gather(
        get_citycode(origin), get_citycode(destination)
    )

    if not citycode_origin:
        raise Exception("City not found for transit origin.")
    if not citycode_destination:
        raise Exception("City not found for transit destination.")

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
        "city1": citycode_origin,
        "city2": citycode_destination,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def direction(
    origin: str, destination: str, mode: str = "driving", waypoints: str | None = None
) -> str:
    """
    Route planning. Supports driving, walking, bicycling, electrobike, transit.

    Args:
        origin: Origin coordinates ("longitude,latitude").
        destination: Destination coordinates ("longitude,latitude").
        mode: Route type. Enum: ["driving", "walking", "bicycling", "electrobike", "transit"].
        waypoints: Via points, semicolon-separated coordinates. Max 16 points.
    """
    if mode == "driving":
        result = await driving_direction(origin, destination, waypoints=waypoints)
    elif mode == "walking":
        result = await walking_direction(origin, destination)
    elif mode == "bicycling":
        result = await bicycling_direction(origin, destination)
    elif mode == "electrobike":
        result = await electrobike_direction(origin, destination)
    elif mode == "transit":
        result = await transit_direction(origin, destination)
    else:
        raise Exception(f"Unsupported mode: {mode}")

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    route = result.get("route")
    if not route:
        raise Exception("No route available.")

    return truncate_text(json2md(route))


# ============================================================================
# Weather
# ============================================================================

@mcp.tool()
async def weather(city: str) -> str:
    """
    Query weather forecast for a city.

    Args:
        city: City name in Chinese (e.g. "北京").
    """
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": AMAP_MAPS_API_KEY,
        "city": city,
        "extensions": "all",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    forecasts = result.get("forecasts")
    if not forecasts:
        raise Exception("No forecast data available.")

    def format_cast(cast):
        return {
            "dayweather": cast["dayweather"],
            "nightweather": cast["nightweather"],
            "daytemp": cast["daytemp"],
            "nighttemp": cast["nighttemp"],
            "daywind": cast["daywind"],
            "nightwind": cast["nightwind"],
            "daypower": cast["daypower"],
            "nightpower": cast["nightpower"],
        }

    def format_forecast(forecast):
        return {
            "city": forecast["city"],
            "province": forecast["province"],
            "casts": [format_cast(cast) for cast in forecast["casts"]],
        }

    forecasts = [format_forecast(forecast) for forecast in forecasts]
    return truncate_text(json2md(forecasts))

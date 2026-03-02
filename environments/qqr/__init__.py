"""QQR-based Travel Planning Evaluation Environment

Uses QQR's MCP tool system for travel planning evaluation.

Problem types:
- InterCity: Inter-city transportation planning
- MultiDay: Multi-day travel planning
- Hybrid: Combined planning (transportation + itinerary)
- SinglePOI: Single point-of-interest exploration
- FoodTour: Food-themed city tour
- Business: Business trip planning
- FamilyStudy: Family study/educational trip

Tools:
- AMap: poi_search, around_search, direction, weather
- Transport: search_flights, search_train_tickets
"""

from .env import Actor
from .problem_generator import ProblemGenerator, TravelProblem, get_generator
from .parser import OutputParser, ParsedOutput, get_parser
from .scorer import TravelScorer, ScoreBreakdown
from .llm_validator import LLMValidator, LLMValidationResult, get_llm_validator

__all__ = [
    "Actor",
    "ProblemGenerator",
    "TravelProblem",
    "get_generator",
    "OutputParser",
    "ParsedOutput",
    "get_parser",
    "TravelScorer",
    "ScoreBreakdown",
    "LLMValidator",
    "LLMValidationResult",
    "get_llm_validator",
]

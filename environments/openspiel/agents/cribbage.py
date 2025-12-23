"""Cribbage Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class CribbageAgent(BaseGameAgent):
    """Cribbage Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "cribbage"
    
    def get_rules(self) -> str:
        return """CRIBBAGE RULES:
Setup: 52-card deck. 2-4 players. Deal 5-6 cards per player. Goal: First to 121 points.
Each player discards cards to form the "crib" (bonus hand for dealer).

Play: Players alternate playing cards. Cannot exceed running total of 31. Score points for:
- Pairs (2 points), runs (1 per card), sum of 15 (2 points), sum of 31 (2 points).

Counting: After play, count hand + starter card for combinations (pairs, runs, 15s, flush).
Dealer also counts the crib.

Scoring is complex; points tracked on cribbage board (pegging)."""

    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Cribbage parameter generation
        """
        players_var = config_id % 3
        return {"players": 2 + players_var}  # 2, 3, 4

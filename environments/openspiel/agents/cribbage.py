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

Play Phase: Players alternate playing cards. Running total cannot exceed 31. Score points for:
- Pairs (2 same rank just played): 2 points
- Three of a kind: 6 points
- Four of a kind: 12 points
- Runs (sequence of 3+ cards): 1 point per card in run
- Sum of 15: 2 points
- Sum of 31: 2 points
- Last card before 31: 1 point

Example Scoring:
- Opponent plays 5, you play 5 → Pair, score 2 points (total is 10)
- Cards 4-5-6 played in sequence → Run of 3, score 3 points
- Cards sum to 15 → Score 2 points

Counting Phase: After play, count hand + starter card for combinations (pairs, runs, 15s, flush).
Dealer also counts the crib.

Card Values: Face cards = 10, Ace = 1, others = face value."""

    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Cribbage parameter generation
        """
        players_var = config_id % 3
        return {"players": 2 + players_var}  # 2, 3, 4

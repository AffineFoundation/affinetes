"""Hearts Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class HeartsAgent(BaseGameAgent):
    """Hearts Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "hearts"
    
    def get_rules(self) -> str:
        return """HEARTS RULES:
Setup: 4 players, 52-card deck. Deal 13 cards each. Goal: Avoid taking hearts and Queen of Spades (penalty cards).

Optional card passing: Before play, pass 3 cards to another player (variant dependent).

Play: Follow suit if possible. Highest card of led suit wins trick. Winner leads next trick.
Cannot lead hearts until hearts "broken" (someone discarded a heart).

Scoring: Each heart = 1 point, Queen of Spades = 13 points. Lowest score wins.
Shooting the moon: Take ALL penalty cards to give 26 points to each opponent instead."""

    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Hearts parameter generation
        """
        variant = config_id % 8
        return {
            "pass_cards": (variant & 1) == 1,
            "jd_bonus": (variant & 2) == 2,
            "avoid_all_tricks_bonus": (variant & 4) == 4
        }

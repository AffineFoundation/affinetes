"""Goofspiel Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class GoofspielAgent(BaseGameAgent):
    """Goofspiel Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "goofspiel"
    
    def get_rules(self) -> str:
        return """GOOFSPIEL RULES:
Setup: Each player has bid cards numbered 1 to N. A prize deck with cards 1 to N is shuffled.
Goal: Win the most points by bidding on prize cards.

Each turn:
1. Reveal top prize card (worth its face value in points)
2. Players simultaneously play one bid card from their hand
3. Highest bidder wins the prize card (adds its value to score)
4. If bids tie, prize card is discarded (no one gets points)

Winning: Player with most points after all rounds wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Goofspiel parameter generation
        """
        cards_var = config_id % 5
        return {
            "players": 2,
            "num_cards": 8 + cards_var * 2,  # 8, 10, 12, 14, 16
            "points_order": "random"
        }

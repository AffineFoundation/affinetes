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
Setup: 4 players, 52-card deck. Deal 13 cards each. Goal: Avoid taking hearts (♥) and Queen of Spades (Q♠).

Optional card passing: Before play, may pass 3 cards to another player (variant dependent).

Play Phase:
1. Must follow the suit that was led if you have that suit
2. If you don't have the led suit, you can play any card
3. Highest card of the LED SUIT wins the trick (not just highest card!)
4. Winner of trick leads next trick
5. Cannot lead hearts until hearts "broken" (someone has discarded a heart on a trick)

Scoring (LOWER is better):
- Each heart card taken: 1 point penalty
- Queen of Spades (Q♠): 13 points penalty
- Your goal: Take the fewest penalty points

Shooting the Moon (advanced): If you take ALL 26 penalty points (all hearts + Q♠), you give 26 points to each opponent instead of taking them yourself.

Card Rank: A (high) > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3 > 2 (low)"""


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
    
    def get_mcts_config(self) -> tuple[int, int]:
        """4 players, 52 cards, 13 tricks. MaxGameLength=64. Complex trick-taking game."""
        return (500, 50)

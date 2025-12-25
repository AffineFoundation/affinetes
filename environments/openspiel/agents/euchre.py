"""Euchre Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class EuchreAgent(BaseGameAgent):
    """Euchre Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "euchre"
    
    def get_rules(self) -> str:
        return """EUCHRE RULES:
Setup: 4 players in 2 teams (N-S vs E-W). 24-card deck (9-A). Deal 5 cards each. Goal: First team to 10 points.
Trump selection: Upcard revealed. Players bid to make it trump or pass. Bidding team must take 3+ tricks.

Play: Follow suit if possible. Highest trump wins, else highest card of led suit.
Card ranking in trump suit: J (right bower) > J of same color (left bower) > A > K > Q > 10 > 9.

Scoring: 3-4 tricks = 1 point, all 5 tricks = 2 points. Going alone (partner sits out): 4 points for all tricks.
Euchred: If bidding team fails to take 3 tricks, opponents score 2 points.

READING THE STATE:
- You see only YOUR hand (shown after "You are Player X")
- Other positions show cards they've played, not their full hands
- S/H/D/C = Spades/Hearts/Diamonds/Clubs
- Card format: rank+suit (e.g., "HA" = Heart Ace, "S9" = Spade 9)
- Tricks section shows what was played in each trick

BIDDING STRATEGY:
- Order up if you have 2+ trump cards and strong support
- Pass if weak hand
- Consider your partner's position

PLAY STRATEGY:
- Lead with trump if you called it
- Follow suit if possible (required)
- Use high cards to win important tricks"""

    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Euchre parameter generation
        """
        variant = config_id % 4
        return {
            "allow_lone_defender": (variant & 1) == 1,
            "stick_the_dealer": (variant & 2) == 2
        }
    
    def get_mcts_config(self) -> tuple[int, int]:
        """4 players, 24 cards, 5 tricks. MaxGameLength=25. Medium complexity trick-taking."""
        return (1000, 100)

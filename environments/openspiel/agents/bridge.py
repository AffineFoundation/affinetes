"""Bridge Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class BridgeAgent(BaseGameAgent):
    """Bridge Game Agent - 4-player card game"""
    
    @property
    def game_name(self) -> str:
        return "bridge"
    
    def get_rules(self) -> str:
        return """BRIDGE RULES:
Players: 4 players in 2 partnerships (North-South vs East-West). 52-card deck.
Goal: Win tricks to fulfill contract bid.

BIDDING PHASE:
- Bid format: Level (1-7) + Strain (♣/♦/♥/♠/NT)
- Level = tricks over 6 required
- Strain ranking: ♣ < ♦ < ♥ < ♠ < NT
- Each bid must be higher than previous
- Special bids: Pass, Double, Redouble
- Bidding ends after 3 consecutive passes

PLAY PHASE:
- Declarer's partner (dummy) reveals all cards
- Declarer controls both hands
- Must follow suit if possible
- Highest card in led suit wins (or highest trump)
- Winner leads next trick

Winning: Fulfill contract (win required tricks). Partnership with most fulfilled contracts wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Bridge parameter generation
        
        Config variants:
        - 0: Standard rubber bridge
        - 1: Chicago bridge (4 deals)
        - 2: Duplicate bridge scoring
        """
        variant = config_id % 3
        
        params = {}
        
        if variant == 0:
            # Rubber bridge (default)
            params = {}
        elif variant == 1:
            # Chicago bridge
            params = {"is_chicago": True}
        else:
            # Duplicate scoring
            params = {"is_duplicate": True}
        
        return params
    
    def get_mcts_config(self) -> tuple[int, int]:
        """
        4-player card game. 52-card deck, complex bidding and play.
        Imperfect information. Bidding phase is fast, play phase is moderate.
        Reduced from (500,50) to balance difficulty vs computation time.
        """
        return (200, 30)
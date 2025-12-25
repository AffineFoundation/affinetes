"""Checkers Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class CheckersAgent(BaseGameAgent):
    """Checkers Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "checkers"
    
    def get_rules(self) -> str:
        return """CHECKERS (DRAUGHTS) RULES:
Board: 8×8 grid, only dark squares used. Each player starts with 12 pieces on dark squares in first 3 rows.
Goal: Capture all opponent's pieces or block them from moving.

Movement: Regular pieces move diagonally forward 1 square. Kings (promoted pieces) move diagonally forward or backward.
Capturing: Jump over adjacent opponent piece to empty square beyond. Multiple jumps allowed in one turn. Capturing is mandatory.

King promotion: When piece reaches opponent's back row, it becomes a King (stack another piece or flip it).
Winning: Capture all opponent pieces or leave opponent with no legal moves."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Checkers parameter generation
        """
        return {}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """8×8 board (dark squares only), mandatory captures. MaxGameLength=1000. High complexity."""
        return (500, 50)

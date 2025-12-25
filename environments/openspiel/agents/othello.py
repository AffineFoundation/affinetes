"""Othello Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class OthelloAgent(BaseGameAgent):
    """Othello Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "othello"
    
    def get_rules(self) -> str:
        return """OTHELLO (REVERSI) RULES:
Board: 8×8 grid. 2 players (Black and White). Start with 4 discs in center (2 black, 2 white diagonal).
Goal: Have more discs of your color when board is full or no moves available.

Turn: Place disc to sandwich opponent's discs between your new disc and existing disc (horizontally, vertically, or diagonally). All sandwiched opponent discs flip to your color.
Must flip at least 1 disc; if no valid move, pass turn.

Winning: Player with most discs when game ends wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Othello parameter generation
        """
        return {}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """8×8 board, 64 actions, MaxGameLength=128. Medium complexity with large branching factor."""
        return (1000, 100)

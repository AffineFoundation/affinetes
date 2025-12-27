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
        """
        8×8 board, 64 actions, MaxGameLength=128. Medium complexity with large branching factor.
        
        Optimization history:
        - Original (500, 50): timeout (>30min)
        - (300, 50): timeout (>30min)
        - (150, 50): 29min (ONLY successful test!)
        - (90, 50): timeout (>30min) - anomaly!
        - (100, 30): timeout (>30min) - anomaly!
        - (80, 20): timeout (>30min) - anomaly!
        
        Analysis: Weaker MCTS → longer games OR API instability
        Strategy: Conservative optimization from proven (150,50)
        Target: 4,800 compute (64% of original, 62% time reduction)
        """
        return (500, 40)

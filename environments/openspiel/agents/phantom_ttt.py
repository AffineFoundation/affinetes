"""Phantom Ttt Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class PhantomTttAgent(BaseGameAgent):
    """Phantom Ttt Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "phantom_ttt"
    
    def get_rules(self) -> str:
        return """PHANTOM TIC-TAC-TOE RULES:
Like standard Tic-Tac-Toe, but with imperfect information: You CANNOT see opponent's moves.
Board: 3×3 grid. Players alternate placing X or O.
Goal: Form 3 in a row (horizontal, vertical, or diagonal).

Key difference: When you try to place on a cell occupied by opponent, move fails silently or you get limited feedback (variant dependent).
Observation variants:
- "reveal-nothing": No feedback on opponent moves
- "reveal-numturns": Know how many turns opponent took

Winning: First to get 3 in a row wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Phantom Ttt parameter generation
        """
        obstype_var = config_id % 2
        obstype_str = "reveal-nothing" if obstype_var == 0 else "reveal-numturns"
        return {"obstype": obstype_str}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """3×3 grid, imperfect information. Simple game with hidden opponent moves."""
        return (3000, 150)

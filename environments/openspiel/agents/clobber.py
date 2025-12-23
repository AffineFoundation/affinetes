"""Clobber Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class ClobberAgent(BaseGameAgent):
    """Clobber Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "clobber"
    
    def get_rules(self) -> str:
        return """CLOBBER RULES:
Board: Rectangular grid (5×5, 6×6, or 7×7) filled with alternating black and white pieces.
Goal: Be the last player able to move.

Movement: On your turn, move one of your pieces orthogonally (horizontally or vertically) to capture an adjacent opponent piece. The captured piece is removed and replaced by your piece.
Must capture: Every move must capture an opponent piece. No non-capturing moves allowed.

Losing: If you have no legal moves (no adjacent opponent pieces to capture), you lose.
Strategy: Force opponent into position with no captures available."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Clobber parameter generation
        """
        size_var = config_id % 3
        board_size = 5 + size_var  # 5, 6, 7
        return {
            "rows": board_size,
            "columns": board_size
        }

"""Breakthrough Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class BreakthroughAgent(BaseGameAgent):
    """Breakthrough Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "breakthrough"
    
    def get_rules(self) -> str:
        return """BREAKTHROUGH RULES:
Board: 6×6 or 8×8 grid. Each player starts with pawns in first 2 rows.
Goal: Move one of your pawns to opponent's back row.

Movement: Pawns move forward one square:
- Straight forward if empty
- Diagonally forward to capture opponent's pawn

Capturing: Mandatory if you choose diagonal move. Captured pawns are removed.

Winning: First player to reach opponent's back row wins, OR capture all opponent pawns."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Breakthrough parameter generation
        """
        size_var = config_id % 2
        return {
            "rows": 6 + size_var * 2,  # 6 or 8
            "columns": 6 + size_var * 2
        }

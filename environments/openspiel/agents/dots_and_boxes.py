"""Dots And Boxes Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class DotsAndBoxesAgent(BaseGameAgent):
    """Dots And Boxes Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "dots_and_boxes"
    
    def get_rules(self) -> str:
        return """DOTS AND BOXES RULES:
Board: Grid of dots (3×3, 4×4, or 5×5). Goal: Complete more boxes than opponent.

Turn: Draw one horizontal or vertical line between two adjacent dots.
Completing a box: When you draw the 4th side of a box, you claim it (mark with your initial) and take another turn.

Scoring: Each completed box = 1 point.
Winning: Player with most completed boxes when all lines drawn wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Dots And Boxes parameter generation
        """
        size_var = config_id % 3
        grid_size = 3 + size_var  # 3, 4, 5
        return {
            "num_rows": grid_size,
            "num_cols": grid_size
        }
    
    def get_mcts_config(self) -> tuple[int, int]:
        """3×3 to 5×5 grid. MaxGameLength varies. Moderate tactical game."""
        return (1200, 100)

"""Quoridor Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class QuoridorAgent(BaseGameAgent):
    """Quoridor Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "quoridor"
    
    def get_rules(self) -> str:
        return """QUORIDOR RULES:
Board: 7×7 or 9×9 grid. 2-4 players, each starts at opposite edge with goal to reach opposite edge.
Goal: Be first to move your pawn to any square on opposite edge.

Each turn, choose ONE action:
- Move pawn: 1 square orthogonally (up/down/left/right). Can jump over adjacent pawn if no wall blocks.
- Place wall: Block 2 spaces between squares (horizontal or vertical). Limited walls per player (8-10).

Wall rules: Cannot completely block any player from reaching their goal edge.
Winning: First pawn to reach target edge wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Quoridor parameter generation
        """
        size_var = (config_id // 2) % 2
        walls_var = config_id % 2
        
        board_size = 7 + size_var * 2  # 7 or 9
        num_walls = 8 + walls_var * 2  # 8 or 10
        
        return {
            "board_size": board_size,
            "wall_count": num_walls
        }
    
    def get_mcts_config(self) -> tuple[int, int]:
        """7×7 or 9×9 board, wall placement strategy. MaxGameLength=4×board². Moderate complexity."""
        return (800, 80)

"""Amazons Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class AmazonsAgent(BaseGameAgent):
    """Amazons Game Agent - Complex territorial control game"""
    
    @property
    def game_name(self) -> str:
        return "amazons"
    
    def get_rules(self) -> str:
        return """AMAZONS RULES:
Board: Grid with 4 amazons per player.
Goal: Be the last player able to move.

Turn structure (2 parts per move):
1. Move one amazon (like chess queen: any distance horizontally/vertically/diagonally)
2. Shoot arrow from new position (also like queen move)

Constraints:
- Cannot move through or onto blocked squares or other amazons
- Arrows permanently block squares
- Must complete both parts if possible
- Lose if no legal move available

Winning: Last player able to move wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Amazons parameter generation
        
        Config variants:
        - 0: 6x6 board (default - balances difficulty and speed)
        - 1: 8x8 board (higher difficulty)
        - 2: 10x10 board (maximum difficulty, may be slow)
        
        Note: Default changed to 6x6 to control computation time
        while maintaining strategic depth.
        """
        size_variant = config_id % 3
        
        if size_variant == 0:
            board_size = 6
        elif size_variant == 1:
            board_size = 8
        else:
            board_size = 10
        
        return {"board_size": board_size}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """
        2-player territorial control. 6x6 default board reduces action space.
        Early game: ~50-100 moves. Late game: ~5-20 moves.
        Reduced from (300,30) to match smaller default board size.
        """
        return (200, 20)
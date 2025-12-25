"""2048 Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class Game2048Agent(BaseGameAgent):
    """2048 Game Agent - Single-player sliding tile puzzle"""
    
    @property
    def game_name(self) -> str:
        return "2048"
    
    def get_rules(self) -> str:
        return """2048 RULES:
Setup: 4x4 grid. Start with two random tiles (2 or 4). Goal: Create a 2048 tile.

Moves: Swipe in 4 directions (Up/Down/Left/Right). All tiles slide in that direction until blocked.
- Tiles with same value merge into one (2+2=4, 4+4=8, etc.)
- Each move spawns a new tile (2 or 4) in a random empty cell
- Score increases by the value of merged tiles

Win condition: Create a 2048 tile (can continue playing after)
Lose condition: Grid is full and no valid moves remain

ACTIONS:
- 0: Up
- 1: Down  
- 2: Left
- 3: Right"""

    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """2048 parameter generation - max_tile configuration"""
        # 2048 only supports max_tile parameter (2048, 4096, 8192, etc.)
        # Default to 2048 for standard gameplay
        return {}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """Single-player 4×4 sliding puzzle. No opponent, MCTS not applicable."""
        return None
"""Hex Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class HexAgent(BaseGameAgent):
    """Hex Game Agent - Enhanced with strategic hints"""
    
    @property
    def game_name(self) -> str:
        return "hex"
    
    def get_rules(self) -> str:
        return """HEX RULES:
Board: Diamond-shaped grid (5×5, 7×7, 9×9, or 11×11). Two players (Red and Blue).
Goal: Connect your two opposite sides of the board with an unbroken chain of your stones.

Turn: Place one stone of your color on any empty cell.
Red (x) connects top-left to bottom-right sides.
Blue (o) connects top-right to bottom-left sides.

Strategy Tips:
- Center positions offer more connection paths
- Block opponent's potential chains
- Build your own chain while defending

No draws possible: Someone must win."""
    
    def format_state(self, state, player_id: int) -> str:
        """Enhanced state formatting with strategic information"""
        base_obs = state.observation_string(player_id)
        
        # Add player role clarity
        if player_id == 0:
            role = "Red (x) - Connect TOP-LEFT to BOTTOM-RIGHT"
        else:
            role = "Blue (o) - Connect TOP-RIGHT to BOTTOM-LEFT"
        
        return f"Current State:\n{base_obs}\n\nYour Role: {role}"
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Hex parameter generation
        """
        size_var = config_id % 4
        board_size = 5 + size_var * 2  # 5, 7, 9, 11
        return {"board_size": board_size}

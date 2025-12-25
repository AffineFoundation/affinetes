"""Pig Dice Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class PigAgent(BaseGameAgent):
    """Pig Dice Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "pig"
    
    def get_rules(self) -> str:
        return """PIG DICE RULES:
Setup: 2 players, 1 six-sided die. Goal: First to reach target score (20-40 points) wins.

Each turn:
- Roll: Roll die and add to turn total. If you roll 1, lose turn total and turn ends.
- Hold: Add turn total to your permanent score and end turn.

Winning: First player to reach or exceed target score wins immediately."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Pig dice game with risk/reward decisions
        
        Config space: 3 variants (different win scores)
        """
        score_var = config_id % 3
        return {
            "players": 2,
            "winscore": 20 + score_var * 10  # 20, 30, 40
        }
    
    def get_mcts_config(self) -> tuple:
        """
        Pig: 2 actions (Roll/Hold), 6-sided die
        MaxGameLength: 1000 but typically 20-40 moves
        Very simple game, can use highest MCTS strength
        Config: (5000, 200)
        """
        return (5000, 200)
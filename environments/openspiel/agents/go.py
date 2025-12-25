"""Go Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class GoAgent(BaseGameAgent):
    """Go Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "go"
    
    def get_rules(self) -> str:
        return """GO RULES:
Board: Grid (7×7, 9×9, or 11×11). 2 players (Black and White). Goal: Control more territory than opponent.

Turn: Place one stone on empty intersection. Once placed, stones don't move (unless captured).
Capture: Surround opponent's stones (remove all liberties/adjacent empty points) to capture and remove them.
Ko rule: Cannot immediately recapture to repeat previous board position.

Territory: Empty intersections surrounded by your stones. Scoring: Territory + captured stones + komi (bonus for White, typically 6.5-7.5 points).
Winning: Higher score wins."""

    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Go parameter generation - Restricting to smaller boards for LLM evaluation
        """
        size_var = config_id % 2
        
        board_size = 5 + size_var * 2  # 5 or 7 (reduced from 7/9/11)
        komi = 7.5  # Standard komi
        
        return {
            "board_size": board_size,
            "komi": komi
        }
    
    def get_mcts_config(self) -> tuple[int, int]:
        """5×5 or 7×7 board. MaxGameLength=2×board². Very high complexity strategy game."""
        return (300, 30)

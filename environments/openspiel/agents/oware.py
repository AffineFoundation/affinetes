"""Oware Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class OwareAgent(BaseGameAgent):
    """Oware Game Agent - African mancala-style strategy game"""
    
    @property
    def game_name(self) -> str:
        return "oware"
    
    def get_rules(self) -> str:
        return """OWARE RULES:
Board: 2 rows of 6 pits. Each player owns one row. Seeds distributed initially.
Goal: Capture more than half of all seeds.

Turn:
1. Pick all seeds from one of your pits (0-5)
2. Sow counter-clockwise, one seed per pit (including opponent's pits)
3. If last seed lands in opponent's pit with 2-3 total seeds, capture them
4. Continue capturing backwards while pits have 2-3 seeds

Constraints:
- Cannot leave opponent with no seeds
- Cannot capture all opponent's seeds at once (Grand Slam)
- Game ends when player cannot move

Winning: Player with most captured seeds wins."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Oware parameter generation
        
        Config variants:
        - 0: Standard rules (4 seeds per pit)
        - 1: Variant with 3 seeds per pit (faster)
        - 2: Variant with 5 seeds per pit (longer games)
        """
        seed_variant = config_id % 3
        
        if seed_variant == 0:
            seeds_per_house = 4  # Standard
        elif seed_variant == 1:
            seeds_per_house = 3  # Faster
        else:
            seeds_per_house = 5  # Longer
        
        return {"seeds_per_house": seeds_per_house}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """
        2-player counting game. Action space: 6 moves per turn.
        Much simpler than Go (5x5) which uses (300,30).
        Reduced from (1000,100) to match complexity level.
        """
        return (300, 30)
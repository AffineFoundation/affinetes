"""Backgammon Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class BackgammonAgent(BaseGameAgent):
    """Backgammon Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "backgammon"
    
    def get_rules(self) -> str:
        return """BACKGAMMON RULES:
Setup: Each player has 15 checkers. Board has 24 points (triangles). Players move in opposite directions.
Goal: Move all your checkers to your home board, then bear them off.

Turn: Roll 2 dice. Move checkers forward by dice values. If doubles, move 4 times.
Movement: Can land on empty point, your own checkers, or single opponent checker (hit it to bar).
Bar: Hit checkers must re-enter from opponent's home board before other moves.

Bearing off: Once all checkers in home board, remove them by rolling their point number or higher.
Winning: First to bear off all checkers wins.

UNDERSTANDING THE BOARD:
- The board display shows a top-down view with 24 points
- 'x' represents your (Player 1's) checkers, 'o' represents opponent's (Player 0's) checkers
- Bar: shows checkers that were hit and must re-enter
- Actions show moves like "8/5/3" meaning move from point 8 to 5 to 3

STRATEGY TIPS:
- Early game: Move checkers forward to establish control
- Avoid leaving single checkers exposed (blots) where possible
- Try to make points (two or more checkers) to block opponent
- When bearing off, remove checkers efficiently"""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Backgammon with dice randomness
        
        Config space: 1 variant (standard rules)
        """
        return {}
    
    def get_mcts_config(self) -> tuple:
        """
        Backgammon: 24 points, 15 checkers/player, 1352 actions
        MaxGameLength: 1500, Complex movement logic + dice randomness
        Config: (1000, 100) - Medium complexity
        """
        return (1000, 100)
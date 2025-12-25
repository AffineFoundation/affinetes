"""Goofspiel Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class GoofspielAgent(BaseGameAgent):
    """Goofspiel Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "goofspiel"
    
    def get_rules(self) -> str:
        return """GOOFSPIEL RULES:
Setup: Each player has bid cards numbered 1 to N. A prize deck with cards 1 to N is shuffled.
Goal: Win the most points by bidding on prize cards.

Each turn:
1. Reveal top prize card (worth its face value in points)
2. Players simultaneously play one bid card from their hand
3. Highest bidder wins the prize card (adds its value to score)
4. If bids tie, prize card is discarded (no one gets points)

Winning: Player with most points after all rounds wins."""
    
    def format_state(self, state, player_id: int) -> str:
        """Format Goofspiel state with better readability"""
        import re
        obs = state.observation_string(player_id)
        
        # Parse and format the points if available
        # Format: "Points: 14 4" -> "Player 0: 14 points, Player 1: 4 points"
        points_match = re.search(r'Points:\s+(\d+)\s+(\d+)', obs)
        if points_match:
            p0_points, p1_points = points_match.groups()
            obs = re.sub(
                r'Points:\s+\d+\s+\d+',
                f'Player 0: {p0_points} points, Player 1: {p1_points} points',
                obs
            )
        
        # Parse and explain win sequence if available
        # Format: "Win sequence: 1 0 0 -3 1" -> explain each number
        win_seq_match = re.search(r'Win sequence:\s+([-\d\s]+)', obs)
        if win_seq_match:
            win_seq = win_seq_match.group(1).strip()
            explanation = "\n(Win sequence: 1=player 1 won, 0=player 0 won, negative=tie)"
            obs = obs.replace(f'Win sequence: {win_seq}', f'Win sequence: {win_seq}{explanation}')
        
        return obs
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Goofspiel parameter generation
        """
        cards_var = config_id % 5
        return {
            "players": 2,
            "num_cards": 8 + cards_var * 2,  # 8, 10, 12, 14, 16
            "points_order": "random"
        }
    
    def get_mcts_config(self) -> tuple:
        """
        Goofspiel is a simultaneous-move game - MCTS not supported.
        env.py automatically uses random bot for such games.
        """
        return None  # Not used

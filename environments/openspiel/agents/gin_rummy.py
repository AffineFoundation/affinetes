"""Gin Rummy Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class GinRummyAgent(BaseGameAgent):
    """Gin Rummy Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "gin_rummy"
    
    def get_rules(self) -> str:
        return """GIN RUMMY RULES:

SETUP:
- 52-card deck, each player receives 7-10 cards (variant dependent)
- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)

MELDS (Valid Combinations):
1. SET: 3+ cards of SAME RANK (e.g., 7♠ 7♥ 7♣)
2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5♦ 6♦ 7♦)
Examples:
- Valid runs: A♠-2♠-3♠, 9♥-10♥-J♥-Q♥, 10♣-J♣-Q♣-K♣
- Invalid: K♠-A♠-2♠ (Ace is LOW only, not wraparound)

CARD NOTATION:
- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)
- Suits: s(spades♠), h(hearts♥), d(diamonds♦), c(clubs♣)
- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades

GAME PHASES:
1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)
2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)
3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)
4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)
5. Knock: Declare end of hand when deadwood ≤ knock_card value

EACH TURN:
1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)
2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)

KNOCKING:
- When deadwood ≤ knock_card value (8-10), you MAY knock to end hand
- Gin: ALL cards form melds (0 deadwood) = 25-point bonus

SCORING: Winner scores difference in deadwood point values.
Card Values: A=1, 2-10=face value, J=11, Q=12, K=13

IMPORTANT: Always respond with the action ID number ONLY, never card names."""
    
    def format_state(self, state, player_id: int) -> str:
        """Format Gin Rummy state - keep original observation_string"""
        return state.observation_string(player_id)
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Gin Rummy parameter generation
        """
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return {
            "hand_size": 7 + hand_var,  # 7, 8, 9
            "knock_card": 10 - knock_var  # 10, 9, 8
        }
    
    def get_mcts_config(self) -> tuple:
        """
        Get MCTS configuration for Gin Rummy
        
        Complexity Analysis:
        - 52-card deck, 7-10 cards per hand
        - State space: >10^85 information states (astronomical!)
        - Opponent hand combinations: 41C10 = 1,121,099,408 possibilities
        - Branching factor: 10-15 actions per turn (discard + meld options)
        - Average game length: 50-100 moves (MaxGameLength = 300)
        - Rollout cost: Very high (complex meld calculations, huge state space)
        
        Token Consumption:
        - From TEST_REPORT: 167.8k avg tokens (highest among all games)
        - Indicates very long games with many interactions
        
        Configuration:
        - max_simulations: 500 (reduced due to high complexity)
        - n_rollouts: 50 (limited due to expensive rollouts)
        
        Time Estimate: ~1200 seconds (20 minutes) for typical game
        Within 30-minute timeout but close to limit.
        
        Returns:
            tuple: (max_simulations, n_rollouts)
        """
        return (500, 50)

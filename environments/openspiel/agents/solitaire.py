"""Solitaire (Klondike) Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class SolitaireAgent(BaseGameAgent):
    """Solitaire Game Agent - Classic Klondike card game"""
    
    @property
    def game_name(self) -> str:
        return "solitaire"
    
    def get_rules(self) -> str:
        return """SOLITAIRE (KLONDIKE) RULES:
Setup: 52-card deck shuffled randomly. 7 tableau columns with 1-7 cards (top card face-up).
Remaining cards form the stock pile (dealt 3 at a time to waste pile).

Goal: Move all cards to 4 foundation piles (one per suit), building from Ace to King.

Card Movement:
1. TABLEAU: Build descending sequences in alternating colors (Red 7 on Black 8)
   - Can move single cards or sequences
   - Empty tableau spaces can only be filled by Kings
   
2. FOUNDATIONS: Build ascending sequences by suit (A♠ → 2♠ → 3♠ ... → K♠)
   - Must start with Ace of each suit
   - Build up in same suit only

3. STOCK/WASTE: Draw 3 cards from stock to waste pile
   - Top waste card is playable
   - Can cycle through stock pile multiple times

WINNING: All 52 cards successfully moved to foundations (sorted by suit, Ace to King)

CARD NOTATION:
- Suits: ♠ (Spades), ♥ (Hearts), ♣ (Clubs), ♦ (Diamonds)
- Ranks: A, 2-10, J, Q, K
- 🂠 = face-down card"""
    
    def format_state(self, state, player_id: int) -> str:
        """Format solitaire game state"""
        try:
            return str(state)
        except:
            return "Unable to get game state"
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """Solitaire parameter generation - standard configuration"""
        # Solitaire doesn't support seed parameter in OpenSpiel
        # Available params: depth_limit, is_colored, players
        # Use default configuration for reproducible gameplay
        return {}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """Single-player card sequencing. No opponent, MCTS not applicable."""
        return None
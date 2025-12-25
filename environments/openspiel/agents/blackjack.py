"""Blackjack Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class BlackjackAgent(BaseGameAgent):
    """Blackjack Game Agent - Custom formatting with point totals"""
    
    @property
    def game_name(self) -> str:
        return "blackjack"
    
    def get_rules(self) -> str:
        return """BLACKJACK RULES:
Setup: 52-card deck. Player vs Dealer. Goal: Get card total closer to 21 than dealer without going over.
Card values: 2-10 = face value, J/Q/K = 10, A = 1 or 11 (player's choice).

Player turn:
- Hit: Take another card
- Stand: Keep current total and end turn

Dealer turn: Must hit on 16 or less, stand on 17 or more.

Winning: Beat dealer's total without exceeding 21. If you exceed 21, you bust and lose immediately."""
    
    def format_state(self, state, player_id: int) -> str:
        """Custom state formatting with point totals calculated"""
        try:
            obs = state.observation_string(player_id)
        except:
            try:
                obs = state.information_state_string(player_id)
            except:
                obs = str(state)
        
        # Calculate and append point totals
        import re
        
        # Extract player cards
        player_match = re.search(r'Player \d+: Cards: ([A-Z0-9 ]+)', obs)
        dealer_match = re.search(r'Dealer: Cards: ([A-Z0-9? ]+)', obs)
        
        result_parts = [obs]
        
        if player_match:
            cards_str = player_match.group(1).strip()
            total = self._calculate_total(cards_str)
            result_parts.append(f"\nYour total: {total} points")
        
        if dealer_match:
            cards_str = dealer_match.group(1).strip()
            # Only calculate if not hidden
            if '?' not in cards_str:
                total = self._calculate_total(cards_str)
                result_parts.append(f"Dealer total: {total} points")
        
        return ''.join(result_parts)
    
    def _calculate_total(self, cards_str: str) -> str:
        """Calculate card total, handling Aces intelligently"""
        cards = cards_str.split()
        total = 0
        aces = 0
        
        for card in cards:
            if card == '??':
                continue
            rank = card[1:]  # Skip suit character
            if rank in ['J', 'Q', 'K']:
                total += 10
            elif rank == 'A':
                aces += 1
                total += 11
            else:
                total += int(rank)
        
        # Adjust for Aces
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return str(total)
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """Blackjack parameter generation - standard configuration"""
        return {}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """Single player vs dealer. MaxGameLength=12. Simple probability game, no MCTS needed."""
        return None

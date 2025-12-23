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

EACH TURN:
1. DRAW: Pick from stock pile OR discard pile (if useful for meld)
2. DISCARD: Place ONE card face-up on discard pile

STRATEGY TIPS:
- Prioritize cards that can form/extend melds
- Discard high-value cards (K, Q, J, 10) if they don't form melds
- Pick from discard pile ONLY if it completes/extends a meld
- Avoid discarding cards that opponent just discarded (they don't need it)
- Keep cards that can combine multiple ways (middle ranks 6-9)

KNOCKING:
- When deadwood ≤ knock_card value (8-10), you MAY knock to end hand
- Gin: ALL cards form melds (0 deadwood) = 25-point bonus

SCORING: Winner scores difference in deadwood point values.
Card Values: A=1, 2-10=face value, J=11, Q=12, K=13

CRITICAL: Don't get stuck in loops! If you just discarded a card, DON'T immediately pick it back up!"""
    
    def format_state(self, state, player_id: int) -> str:
        """Format Gin Rummy state for better LLM understanding"""
        state_str = state.observation_string(player_id)
        lines = state_str.split('\n')
        
        # Extract key information
        knock_card = None
        prev_upcard = None
        repeated_move = 0
        phase = None
        your_hand = []
        
        for line in lines:
            if line.startswith("Knock card:"):
                knock_card = line.split(":")[1].strip()
            elif line.startswith("Prev upcard:"):
                prev_upcard = line.split(":")[1].strip()
            elif line.startswith("Repeated move:"):
                repeated_move = int(line.split(":")[1].strip())
            elif line.startswith("Phase:"):
                phase = line.split(":")[1].strip()
            elif line.startswith("Player0:") or line.startswith("Player1:"):
                # Extract deadwood value
                if "Deadwood=" in line:
                    deadwood = line.split("Deadwood=")[1].strip()
        
        # Build formatted state
        result = f"\n=== GAME STATE ===\n"
        result += f"Phase: {phase}\n"
        result += f"Knock Threshold: {knock_card} (can knock when deadwood ≤ {knock_card})\n"
        
        if prev_upcard and prev_upcard != "XX":
            result += f"\n⚠️ LAST DISCARDED CARD: {prev_upcard}\n"
            result += f"   (Avoid picking this back up immediately!)\n"
        
        if repeated_move > 0:
            result += f"\n⚠️⚠️ WARNING: Repeated move detected! ({repeated_move})\n"
            result += f"   You're in a loop! Change your strategy NOW!\n"
        
        result += f"\n{state_str}\n"
        
        result += f"\n=== DECISION HINTS ===\n"
        if phase == "Draw":
            result += "- Drawing from STOCK: Get unknown card (safer)\n"
            result += "- Drawing from DISCARD: Only if it helps form/complete a meld!\n"
            result += "  Ask yourself: Does this card complete a set or run?\n"
        elif phase == "Discard":
            result += "- Discard HIGH cards (K, Q, J, 10) that don't form melds\n"
            result += "- Keep cards that can form multiple combinations\n"
            result += "- DON'T discard cards you just picked up (causes loops!)\n"
        
        return result
    
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

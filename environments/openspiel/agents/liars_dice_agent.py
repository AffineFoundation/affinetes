"""Liar's Dice Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
import pyspiel
from typing import Dict, Any


class LiarsDiceAgent(BaseGameAgent):
    """Liar's Dice Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "liars_dice"
    
    def get_rules(self) -> str:
        return """LIAR'S DICE RULES:

Setup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.

Goal: Make bids about total dice across ALL players, or call "Liar" on opponent's bid.

Actions:
- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.
- Call Liar: Challenge the previous bid.

Bidding rules: Each bid must be higher than the previous bid. "Higher" means:
  - Same face value but higher quantity (e.g., "2 fours" beats "1 four")
  - Same quantity but higher face value (e.g., "2 fives" beats "2 fours")

Wild dice: 6s are WILD and count as ANY face value.
- When counting dice for a bid, include 6s in the count
- Example: Bid "3 fours" means at least 3 dice showing EITHER 4 OR 6

Winning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.
"""
    
    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """
        Format Liar's Dice state
        
        Convert OpenSpiel's compact state (e.g., "36 45") to LLM-friendly description
        """
        # 1. Extract player dice
        player_dice = self._extract_player_dice(state, player_id)
        
        # 2. Get game configuration
        num_players = state.num_players()
        num_dice_per_player = self._get_num_dice_per_player(state, player_id)
        total_dice = num_dice_per_player * num_players
        
        # 3. Extract current bid (if any)
        current_bid = self._extract_current_bid(state)
        
        # 4. Build state description
        state_parts = [
            f"Your dice: {player_dice}",
            f"Number of dice per player: {num_dice_per_player}",
            f"Total dice in game: {total_dice}",
            f"Number of players: {num_players}",
            f"Current player to act: Player {state.current_player()}"
        ]
        
        if current_bid:
            state_parts.append(f"\nCurrent bid: {current_bid}")
            state_parts.append("You can either: (1) Make a higher bid, or (2) Call 'Liar'")
        else:
            state_parts.append("No bid yet - you must make the first bid")
        
        return "\n".join(state_parts)
    
    def _extract_player_dice(self, state: pyspiel.State, player_id: int) -> str:
        """
        Extract player dice from information state string
        
        Information state format examples:
        - "36" -> dice are [3, 6]
        - "1 1-3" -> die is [1], bid history is "1-3"
        - "36 1-3 2-5" -> dice are [3, 6], bid history is "1-3 2-5"
        
        Strategy: Extract only the first group of consecutive digits before any space
        """
        try:
            info_str = state.information_state_string(player_id)
            
            if not info_str:
                return "[Unable to determine - empty info string]"
            
            # Split by space and take only the first part (player's dice)
            # "1 1-3 2-5" -> "1"
            # "36 1-3" -> "36"
            # "145" -> "145"
            first_part = info_str.split()[0] if ' ' in info_str else info_str
            
            # Extract only digits from the first part
            dice = [int(d) for d in first_part if d.isdigit()]
            
            if dice:
                return f"{dice} (showing: {', '.join(map(str, dice))})"
            else:
                return "[Unable to determine - no digits in first part]"
        except Exception as e:
            return f"[Error extracting dice: {e}]"
    
    def _extract_current_bid(self, state: pyspiel.State) -> str:
        """
        Extract current bid from information state
        
        Parse the last bid from information state string
        Format: "dice_values bid_history"
        Example: "36 1-3 2-5" means bids were "1-3", then "2-5"
        """
        try:
            player_id = state.current_player()
            if player_id < 0:
                return None
                
            # Get information state for current player
            info_str = state.information_state_string(player_id)
            
            if not info_str or ' ' not in info_str:
                return None
            
            # Split by space: first part is dice, rest are bids
            parts = info_str.split()
            
            # Find bid history (format: "quantity-face")
            bids = [p for p in parts[1:] if '-' in p]
            
            if bids:
                last_bid = bids[-1]
                quantity, face = last_bid.split('-')
                return f'"{quantity}-{face}" (claiming at least {quantity} dice showing {face} across all players)'
            else:
                return None
        except Exception as e:
            return None
    
    def _extract_bid_details(self, bid_str: str) -> tuple:
        """Extract quantity and face from bid string like '"2-3" (claiming...)' -> (2, '3')"""
        try:
            # Extract the bid part "2-3" from the full description
            if '"' in bid_str:
                bid_part = bid_str.split('"')[1]  # "2-3"
                quantity, face = bid_part.split('-')  # "2", "3"
                return (int(quantity), face)
            return (None, None)
        except:
            return (None, None)
    
    def _count_matching_dice(self, dice_str: str, face: str) -> int:
        """Count how many dice match the bid face (including 6s as wild)"""
        try:
            # Extract dice values from string like "[3, 6] (showing: 3, 6)"
            if '[' in dice_str and ']' in dice_str:
                dice_part = dice_str.split('[')[1].split(']')[0]
                dice = [d.strip() for d in dice_part.split(',')]
                
                # Count matches: either the exact face or 6 (wild)
                count = sum(1 for d in dice if d == face or d == '6')
                return count
            return 0
        except:
            return 0
    
    def _get_num_dice_per_player(self, state: pyspiel.State, player_id: int) -> int:
        """Get number of dice per player from information state"""
        try:
            info_str = state.information_state_string(player_id)
            
            if not info_str:
                return 2
            
            # Split by space and take only the first part (player's dice)
            first_part = info_str.split()[0] if ' ' in info_str else info_str
            
            # Number of dice = number of digit characters in first part
            num_dice = len([c for c in first_part if c.isdigit()])
            return num_dice if num_dice > 0 else 2
        except:
            return 2  # Fallback
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Generate Liar's Dice parameters
        """
        return {
            "players": 2,
            "numdice": 5
        }
    
    def get_mcts_config(self) -> tuple:
        """
        Get MCTS configuration for Liar's Dice
        
        Complexity Analysis:
        - 2 players, 5 dice each (10 total)
        - Branching factor: ~15-30 bid options per turn
        - Average game length: 5-10 moves
        - Rollout cost: Very low (simple state, no complex calculations)
        
        Configuration:
        - max_simulations: 5000 (very high, game is simple enough)
        - n_rollouts: 200 (extensive evaluation for maximum strength)
        
        Returns:
            tuple: (max_simulations, n_rollouts)
        """
        return (5000, 200)
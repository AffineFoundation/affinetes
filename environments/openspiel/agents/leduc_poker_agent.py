"""Leduc Poker Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
import pyspiel
from typing import Dict, Any
import re


class LeducPokerAgent(BaseGameAgent):
    """Leduc Poker Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "leduc_poker"
    
    def get_rules(self) -> str:
        return """LEDUC POKER RULES:

Deck: 2 suits × (num_players + 1) ranks. For 2 players: 6 cards (J♠ J♥ Q♠ Q♥ K♠ K♥).

Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.

Round 1: Each player receives one private card. Actions: Fold (lose ante), Call/Check (match current bet), Raise (add 2 chips to bet). Maximum 2 raises per round.
Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.

Winning: Player with best hand wins pot (or last remaining if others fold).
Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).
"""
    
    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """
        Format Leduc Poker state
        
        Convert OpenSpiel's information state string to LLM-friendly description
        
        Format: [Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]
        """
        try:
            info_str = state.information_state_string(player_id)
            
            # Parse information state string using regex
            private_card = self._extract_field(info_str, r'\[Private: (-?\d+)\]')
            round_num = self._extract_field(info_str, r'\[Round (\d+)\]')
            pot = self._extract_field(info_str, r'\[Pot: (\d+)\]')
            money = self._extract_field(info_str, r'\[Money: ([\d ]+)\]')
            public_card = self._extract_field(info_str, r'\[Public: (-?\d+)\]')
            round1_seq = self._extract_field(info_str, r'\[Round1: ([^\]]*)\]')
            round2_seq = self._extract_field(info_str, r'\[Round2: ([^\]]*)\]')
            current_player = self._extract_field(info_str, r'\[Player: (-?\d+)\]')
            
            # Build state description
            state_parts = []
            
            # 1. Your card
            if private_card and private_card != "-10000":
                card_name = self._card_id_to_name(int(private_card), state.num_players())
                state_parts.append(f"Your card: {card_name}")
            else:
                state_parts.append("Your card: (not dealt yet)")
            
            # 2. Public card (if round 2)
            if public_card and public_card != "-10000":
                card_name = self._card_id_to_name(int(public_card), state.num_players())
                state_parts.append(f"Public card: {card_name}")
                
                # Check for pair
                if private_card and private_card != "-10000":
                    if self._is_pair(int(private_card), int(public_card)):
                        state_parts.append("Hand: Pair")
            
            # 3. Round info
            state_parts.append(f"Current round: {round_num}/2")
            state_parts.append(f"Pot size: {pot} chips")
            
            # 4. Money
            if money:
                money_list = money.split()
                if len(money_list) >= 2:
                    state_parts.append(f"Your chips: {money_list[player_id]}")
                    opponent_idx = 1 - player_id
                    state_parts.append(f"Opponent chips: {money_list[opponent_idx]}")
            
            # 5. Betting history
            if round1_seq:
                betting_desc = self._parse_betting_sequence(round1_seq)
                state_parts.append(f"Round 1 betting: {betting_desc}")
            
            if round2_seq:
                betting_desc = self._parse_betting_sequence(round2_seq)
                state_parts.append(f"Round 2 betting: {betting_desc}")
            
            # 6. Current player
            if current_player and current_player != "-1":
                cp = int(current_player)
                if cp == player_id:
                    state_parts.append("Your turn to act")
                else:
                    state_parts.append("Waiting for opponent")
            
            return "\n".join(state_parts)
            
        except Exception as e:
            return f"[Error formatting state: {e}]\nRaw info state: {state.information_state_string(player_id)}"
    
    def _extract_field(self, info_str: str, pattern: str) -> str:
        """Extract field using regex"""
        match = re.search(pattern, info_str)
        return match.group(1) if match else ""
    
    def _card_id_to_name(self, card_id: int, num_players: int) -> str:
        """
        Convert card ID to human-readable name
        
        2-player (6 cards):
          0=J♠, 1=J♥, 2=Q♠, 3=Q♥, 4=K♠, 5=K♥
        3-player (8 cards):
          0=J♠, 1=J♥, 2=Q♠, 3=Q♥, 4=K♠, 5=K♥, 6=A♠, 7=A♥
        """
        ranks = ['J', 'Q', 'K', 'A']
        suits = ['♠', '♥']
        
        rank_idx = card_id // 2
        suit_idx = card_id % 2
        
        if rank_idx < len(ranks):
            return f"{ranks[rank_idx]}{suits[suit_idx]}"
        else:
            return f"Card_{card_id}"
    
    def _is_pair(self, private_card: int, public_card: int) -> bool:
        """Check if private and public cards form a pair"""
        # Cards form a pair if they have the same rank (same rank_idx)
        private_rank = private_card // 2
        public_rank = public_card // 2
        return private_rank == public_rank
    
    def _parse_betting_sequence(self, sequence_str: str) -> str:
        """
        Parse betting sequence
        
        Input: "1 2 1" or ""
        Output: "Call, Raise, Call" or "(no actions yet)"
        """
        actions_map = {0: "Fold", 1: "Call", 2: "Raise"}
        
        if not sequence_str or sequence_str.strip() == "":
            return "(no actions yet)"
        
        # Extract numbers
        numbers = [int(x) for x in sequence_str.split() if x.isdigit()]
        
        if not numbers:
            return "(no actions yet)"
        
        return ", ".join(actions_map.get(a, f"Action{a}") for a in numbers)
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Generate Leduc Poker parameters
        
        Config space: 1 variant (2-player standard rules)
        """
        return {"players": 2}
    
    def get_mcts_config(self) -> tuple:
        """
        Get MCTS configuration for Leduc Poker
        
        Complexity Analysis:
        - 2 players, 6-card deck (J/Q/K × 2 suits)
        - Branching factor: 3 (Fold/Call/Raise only)
        - Average game length: 4-8 moves (2 rounds × max 4 bets/round)
        - MaxGameLength: 8 moves
        - Rollout cost: Very low (936 total info states, simple logic)
        
        Benchmark Results:
        - 5000×200 config: 18.4 seconds for 20 games
        - Extremely fast due to small action space (3 actions)
        
        Configuration:
        - max_simulations: 5000 (very high, game is simple)
        - n_rollouts: 200 (extensive evaluation for maximum strength)
        
        Time Estimate: ~500 seconds (8.3 minutes) for typical game
        Well within 30-minute timeout.
        
        Returns:
            tuple: (max_simulations, n_rollouts)
        """
        return (5000, 200)
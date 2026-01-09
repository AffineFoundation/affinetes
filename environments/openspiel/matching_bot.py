"""Matching Bot for Goofspiel - bids the value of the prize card"""

import re
import pyspiel
import numpy as np
from typing import Optional, Dict, List, Any

from base_agent import BaseGameAgent


class MatchingBot(pyspiel.Bot):
    """
    Deterministic bot for Goofspiel that always bids the value of the prize card.
    
    Strategy: Match the prize card value with your bid card.
    This is optimal against random opponents and a strong baseline strategy.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        agent: BaseGameAgent,
        player_id: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Matching Bot
        
        Args:
            game: pyspiel.Game instance (should be goofspiel)
            player_id: Player ID (0 or 1)
            agent: BaseGameAgent for game-specific formatting
            seed: Random seed (for fallback random selection)
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._agent = agent
        self._seed = seed or 42
        self._rng = np.random.RandomState(self._seed)
        
        # Conversation history (LLM format for training)
        self._conversation: List[Dict[str, str]] = []
        self._action_history: List[Dict[str, Any]] = []
        self._system_prompt_generated = False
        self._observation: Optional[str] = None

    def restart_at(self, state):
        """Reset for new game"""
        self._conversation.clear()
        self._action_history.clear()
        self._system_prompt_generated = False
        self._observation = None

    def inform_action(self, state, player_id, action):
        """Record all players' actions"""
        try:
            action_str = state.action_to_string(player_id, action)
        except:
            action_str = str(action)
        
        self._action_history.append({
            "player_id": int(player_id),
            "action": int(action),
            "action_str": action_str,
            "is_algorithm": bool(player_id == self._player_id)
        })

        try:
            self._observation = state.observation_string()
        except:
            try:
                self._observation = str(state)
            except:
                self._observation = None

    def _extract_prize_card(self, state) -> Optional[int]:
        """Extract the current prize card value from state"""
        try:
            obs = state.observation_string(self._player_id)
            # Look for "Point card: X" pattern
            match = re.search(r'Point card:\s*(\d+)', obs)
            if match:
                return int(match.group(1))
            
            # Alternative: look for "Current point card: X"
            match = re.search(r'Current point card:\s*(\d+)', obs)
            if match:
                return int(match.group(1))
                
            # Try state string
            state_str = str(state)
            match = re.search(r'point card[:\s]+(\d+)', state_str, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except:
            pass
        return None

    def step(self, state) -> int:
        """
        Get action: bid the prize card value if possible.
        
        Generates prompts identical to LLMBot for training data consistency.
        """
        # Generate system prompt (first time only)
        if not self._system_prompt_generated:
            system_prompt = self._agent.generate_system_prompt()
            self._conversation.append({"role": "system", "content": system_prompt})
            self._system_prompt_generated = True

        # Get legal actions
        legal_actions = state.legal_actions(self._player_id)

        # Generate user prompt (same as LLMBot)
        user_prompt = self._agent.generate_user_prompt(
            state=state,
            player_id=self._player_id,
            legal_actions=legal_actions
        )
        self._conversation.append({"role": "user", "content": user_prompt})

        # Matching strategy: bid the prize card value
        prize_card = self._extract_prize_card(state)
        
        action = None
        if prize_card is not None:
            # In goofspiel, action IDs typically correspond to bid values (0-indexed or 1-indexed)
            # Try exact match first (0-indexed: action = prize_card - 1)
            if prize_card - 1 in legal_actions:
                action = prize_card - 1
            # Try 0-indexed match
            elif prize_card in legal_actions:
                action = prize_card
        
        # Fallback: random if no match found
        if action is None or action not in legal_actions:
            action = self._rng.choice(legal_actions)

        # Record as assistant response
        self._conversation.append({"role": "assistant", "content": str(action)})

        # Record in action history
        self.inform_action(state, self._player_id, action)

        return action

    def get_conversation(self) -> List[Dict[str, str]]:
        """Return conversation history in LLM format for training"""
        return self._conversation.copy()

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Return action history"""
        return self._action_history.copy()

    def get_observation(self) -> Optional[str]:
        """Return last observation"""
        return self._observation

    def get_algorithm(self) -> str:
        """Return algorithm name"""
        return "matching"


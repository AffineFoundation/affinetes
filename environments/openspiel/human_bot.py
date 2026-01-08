"""Human Bot for CLI interaction with OpenSpiel games"""

import pyspiel
import numpy as np
from typing import Optional, Dict, List, Any

from base_agent import BaseGameAgent


class HumanBot(pyspiel.Bot):
    """
    Human player bot for CLI interaction.
    
    Displays game state and legal actions, prompts for input,
    validates input, and records conversation history in LLM format.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        agent: BaseGameAgent,
        seed: Optional[int] = None,
    ):
        """
        Initialize Human Bot
        
        Args:
            game: pyspiel.Game instance
            player_id: Player ID (0, 1, ...)
            agent: BaseGameAgent for game-specific formatting
            seed: Random seed (unused, for interface consistency)
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._agent = agent
        self._seed = seed
        
        # Conversation history (LLM format)
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
            "is_human": bool(player_id == self._player_id)
        })

        try:
            self._observation = state.observation_string()
        except:
            try:
                self._observation = str(state)
            except:
                self._observation = None

    def step(self, state) -> int:
        """
        Prompt human for action via CLI
        
        Displays state, legal actions, and validates input.
        Records interaction in conversation format.
        """
        # Generate system prompt (first time only)
        if not self._system_prompt_generated:
            system_prompt = self._agent.generate_system_prompt()
            self._conversation.append({"role": "system", "content": system_prompt})
            self._system_prompt_generated = True
            print("\n" + "="*60)
            print("GAME RULES:")
            print("="*60)
            print(self._agent.get_rules())
            print("="*60 + "\n")
        
        # Get legal actions
        legal_actions = state.legal_actions(self._player_id)
        
        # Generate and display user prompt
        user_prompt = self._agent.generate_user_prompt(
            state=state,
            player_id=self._player_id,
            legal_actions=legal_actions
        )
        self._conversation.append({"role": "user", "content": user_prompt})
        
        # Display to human
        print("\n" + "-"*40)
        print(user_prompt)
        print("-"*40)
        
        # Build action lookup for validation
        action_map = {str(a): a for a in legal_actions}
        
        # Input loop with validation
        while True:
            try:
                user_input = input("Enter action ID: ").strip()
                
                if user_input in action_map:
                    action = action_map[user_input]
                    # Record as assistant response (same format as LLM)
                    self._conversation.append({"role": "assistant", "content": user_input})
                    self.inform_action(state, self._player_id, action)
                    return action
                else:
                    print(f"Invalid action '{user_input}'. Please enter a valid action ID.")
                    
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                raise
            except EOFError:
                print("\nNo input available.")
                raise

    def get_conversation(self) -> List[Dict[str, str]]:
        """Return conversation history in LLM format"""
        return self._conversation.copy()

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Return action history"""
        return self._action_history.copy()

    def get_observation(self) -> Optional[str]:
        """Return last observation"""
        return self._observation


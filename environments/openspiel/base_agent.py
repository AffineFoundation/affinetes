"""Base Game Agent for OpenSpiel games"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pyspiel


class BaseGameAgent(ABC):
    """
    Base class for game-specific agents.
    
    Each game implements an Agent subclass that encapsulates:
    - State formatting
    - Rule descriptions
    - Prompt generation
    - Parameter configuration
    """
    
    @property
    @abstractmethod
    def game_name(self) -> str:
        """Game name in OpenSpiel"""
        pass
    
    @abstractmethod
    def get_rules(self) -> str:
        """
        Return game rules text
        
        Returns:
            Complete rule description text
        """
        pass

    def format_state(self, state, player_id: int) -> str:
        try:
            return state.observation_string(player_id)
        except:
            try:
                return state.information_state_string(player_id)
            except:
                return str(state)

    @abstractmethod
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Generate game parameters based on config_id
        
        Args:
            config_id: Configuration variant ID (0-99999999)
            
        Returns:
            Game parameter dictionary
        """
        pass

    
    def generate_system_prompt(self) -> str:
        """
        Generate system prompt (called once per game)
        
        Includes: game name, game rules, output format requirements
        
        Returns:
            System prompt text
        """
        rules = self.get_rules()
        
        parts = [
            f"You are playing {self.game_name}.",
        ]
        
        if rules:
            parts.append(f"\n# Game Rules\n{rules}\n")
        
        parts.extend([
            "\n# Output Format",
            "You must respond with ONLY the action ID (a single number).",
            "Do NOT include descriptions or explanations.",
            "\nExamples:",
            '- For action "0 -> roll": respond "0"',
            '- For action "89 -> a3": respond "89"',
        ])
        
        return "\n".join(parts)
    
    def generate_user_prompt(
        self,
        state: pyspiel.State,
        player_id: int,
        legal_actions: List[int]
    ) -> str:
        """
        Generate user prompt (called each turn)
        
        Includes: current state, legal actions
        
        Args:
            state: Game state
            player_id: Player ID
            legal_actions: Pre-computed legal actions
            
        Returns:
            User prompt text
        """
        # 1. Format state
        state_desc = self.format_state(state, player_id)
        
        actions_desc = []
        for action in legal_actions:
            try:
                action_str = state.action_to_string(player_id, action)
                actions_desc.append(f"{action} -> {action_str}")
            except Exception as e:
                actions_desc.append(f"{action}")
        
        # 3. Build prompt (NO action history - LLM has full conversation history)
        prompt_parts = [
            f"Current State:\n{state_desc}\n",
            f"\nYou are Player {player_id}.\n",
            f"Legal Actions:\n" + "\n".join(actions_desc) + "\n\n",
            "Your choice (ID only):"
        ]
        
        return "".join(prompt_parts)
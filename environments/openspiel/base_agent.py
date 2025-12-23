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
    
    def format_action_history(
        self,
        action_history: List[Tuple[int, int]],
        state: pyspiel.State
    ) -> str:
        """
        Format action history (default implementation)
        
        Args:
            action_history: List of (player_id, action) tuples
            state: Current game state
            
        Returns:
            Formatted action history
        """
        if not action_history:
            return "No actions taken yet."
        
        # For games with moves_history (like chess), use it to avoid state-dependent errors
        if hasattr(state, 'moves_history'):
            try:
                moves = state.moves_history()
                if moves and len(moves) == len(action_history):
                    lines = []
                    for i, (player_id, _) in enumerate(action_history):
                        move = moves[i]
                        if hasattr(move, 'to_string'):
                            move_str = move.to_string()
                        else:
                            move_str = str(move)
                        lines.append(f"Player {player_id}: {move_str}")
                    return "\n".join(lines)
            except:
                pass
        
        # Fallback: simple action ID list (avoids action_to_string errors)
        lines = []
        for player_id, action in action_history:
            lines.append(f"Player {player_id}: action {action}")
        
        return "\n".join(lines)
    
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
        action_history: List[Tuple[int, int]]
    ) -> str:
        """
        Generate user prompt (called each turn)
        
        Includes: current state, legal actions
        
        NOTE: Action history is NOT included here because LLM already maintains
        full conversation history. Including it would be redundant and waste tokens.
        
        Args:
            state: Game state
            player_id: Player ID
            action_history: Action history (unused, kept for compatibility)
            
        Returns:
            User prompt text
        """
        # 1. Format state
        state_desc = self.format_state(state, player_id)
        
        # 2. Legal actions
        legal_actions = state.legal_actions(player_id)
        actions_desc = [
            f"{action} -> {state.action_to_string(player_id, action)}"
            for action in legal_actions
        ]
        
        # 3. Build prompt (NO action history - LLM has full conversation history)
        prompt_parts = [
            f"Current State:\n{state_desc}\n",
            f"\nYou are Player {player_id}.\n",
            f"Legal Actions:\n" + "\n".join(actions_desc) + "\n\n",
            "Your choice (ID only):"
        ]
        
        return "".join(prompt_parts)
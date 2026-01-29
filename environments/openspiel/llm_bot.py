"""LLM Bot implementation for OpenSpiel with conversation history support"""

import os
import pyspiel
import numpy as np
import asyncio
import re
import concurrent.futures
from typing import Tuple, Optional, Dict, List, Any

from base_agent import BaseGameAgent
from affinetes.core.llm_chat import llm_chat

# Constants
DEFAULT_MAX_PARSING_RETRIES = 2


class APIError(Exception):
    """Raised when LLM API call fails after all retry attempts"""
    pass


class ParsingError(Exception):
    """Raised when action parsing fails after all retry attempts"""
    pass


class LLMBot(pyspiel.Bot):
    """
    Wraps LLM as an OpenSpiel Bot with conversation history management
    
    This implementation maintains full conversation history and supports
    retry mechanism with context-aware error feedback.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        base_url: str,
        api_key: str,
        model: str,
        temperature: Optional[float],
        rng_seed: int,
        agent: BaseGameAgent,
        seed: Optional[int] = None,
        max_parsing_retries: int = DEFAULT_MAX_PARSING_RETRIES,
        executor: concurrent.futures.ThreadPoolExecutor = None,
    ):
        """
        Initialize LLM Bot with conversation history support

        Args:
            game: pyspiel.Game instance
            player_id: Player ID (0 or 1)
            base_url: LLM API base URL
            api_key: API authentication key
            model: Model name
            temperature: Sampling temperature (None = use model default)
            rng_seed: Random seed for fallback action selection
            agent: BaseGameAgent for game-specific logic (REQUIRED)
            seed: Random seed for LLM API reproducibility
            max_parsing_retries: Maximum parsing retry attempts
            executor: Shared ThreadPoolExecutor for concurrent LLM calls
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._rng = np.random.RandomState(rng_seed)
        self._max_parsing_retries = max_parsing_retries
        self._agent = agent
        self._executor = executor

        self._conversation: List[Dict[str, str]] = []
        self._action_history: List[Dict[str, Any]] = []
        self._system_prompt_generated = False
        self._last_error: Optional[str] = None
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._observation: Optional[str] = None

    def restart_at(self, state):
        """Reset to new game"""
        self._conversation.clear()
        self._action_history.clear()
        self._system_prompt_generated = False
        self._last_error = None
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._observation = None

    def inform_action(self, state, player_id, action):
        """Record all players' actions for game replay and verification"""
        try:
            action_str = state.action_to_string(player_id, action)
        except:
            action_str = str(action)
        
        # Convert numpy types to Python native types for JSON serialization
        self._action_history.append({
            "player_id": int(player_id),
            "action": int(action),
            "action_str": action_str,
            "is_llm": bool(player_id == self._player_id)
        })

        try:
            self._observation = state.observation_string(self._player_id)
        except:
            try:
                self._observation = str(state)
            except:
                self._observation = None

    def step(self, state):
        """
        Core method: choose action with conversation history and retry mechanism

        This is called by evaluate_bots during game play.
        """
        # Generate system prompt (first time only)
        if not self._system_prompt_generated:
            system_prompt = self._agent.generate_system_prompt()
            self._conversation.append({"role": "system", "content": system_prompt})
            self._system_prompt_generated = True
        
        # Get legal actions ONCE at the start of this turn
        legal_actions = state.legal_actions(self._player_id)
        
        # Generate user prompt
        user_prompt = self._agent.generate_user_prompt(
            state=state,
            player_id=self._player_id,
            legal_actions=legal_actions
        )
        self._conversation.append({"role": "user", "content": user_prompt})
        
        # Retry loop for parsing
        for attempt in range(self._max_parsing_retries + 1):
            try:
                # Call LLM API with full conversation history
                response, usage = self._call_llm_api()
            except Exception as e:
                # API errors (timeout, rate limit, etc.) are handled by OpenAI SDK
                # Record error and re-raise
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                self._last_error = f"[API_ERROR] {error_msg}"
                raise APIError(f"LLM API call failed: {error_msg}")
            
            self._conversation.append({"role": "assistant", "content": response})
            
            # Parse action using the SAME legal_actions from the prompt
            result = self._parse_action(response, state, legal_actions)
            
            if result['success']:
                # Success: record action and return
                action = result['action']
                self.inform_action(state, self._player_id, action)
                return action
            
            # Parsing failed - use simplified error message to avoid response contamination
            error_msg = (
                f"Invalid response format. "
                f"You must respond with ONLY the action ID number (e.g., '5'). "
                f"This is attempt {attempt + 1} of {self._max_parsing_retries + 1}."
            )
            self._conversation.append({"role": "user", "content": error_msg})
            if attempt >= self._max_parsing_retries:
                raise ParsingError(
                    f"Failed to parse valid action after {self._max_parsing_retries + 1} retries. "
                    f"Last response: '{response}'. Error: {result['error_message']}"
                )
        
        raise RuntimeError("Should not reach here")


    def _call_llm_api(self) -> Tuple[str, Dict]:
        """Call LLM API via shared llm_chat helper with streaming + retries"""
        async def _run():
            result = await llm_chat(
                messages=self._conversation,
                model=self._model,
                base_url=self._base_url,
                api_key=self._api_key,
                temperature=self._temperature,
                seed=self._seed,
                stream=True,
                max_retries=10,
                strip_think_tags=True,
            )
            return result.content or "", result.usage

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_run())
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                return result
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        # Execute async call in thread pool
        if self._executor:
            result = self._executor.submit(run_async).result()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(run_async).result()

        response, usage = result
        if usage:
            self._total_usage = usage.copy()

        return response, usage

    def _parse_action(self, response: str, state, legal_actions: List[int]) -> Dict:
        """
        Robust action parsing with multiple strategies
        
        Returns dict with keys: success, action, error_message, matched_method
        """
        response_clean = response.strip()
        
        # Strategy 1: Pure number (highest priority)
        if match := re.search(r'^\s*(\d+)\s*$', response_clean):
            try:
                action = int(match.group(1))
                if action in legal_actions:
                    return {'success': True, 'action': action, 'error_message': '', 'matched_method': 'pure_number'}
                else:
                    return {
                        'success': False,
                        'action': None,
                        'error_message': f"Number {action} not in legal actions: {legal_actions}",
                        'matched_method': 'number_invalid'
                    }
            except ValueError as e:
                return {
                    'success': False,
                    'action': None,
                    'error_message': f"Cannot convert to integer: {str(e)}. Model generated invalid action.",
                    'matched_method': 'number_conversion_error'
                }
        
        # Strategy 2: Find legal action ID in text
        for action in legal_actions:
            if re.search(rf'\b{action}\b', response_clean):
                return {'success': True, 'action': action, 'error_message': '', 'matched_method': 'number_in_text'}
        
        # Strategy 3: Match action string (exact or simplified)
        action_map = self._build_action_string_map(state, legal_actions)
        response_lower = response_clean.lower()
        response_simplified = re.sub(r'[^a-z0-9]', '', response_lower)
        
        # Try exact match first, then simplified
        for action_str, action_id in action_map.items():
            if action_str in response_lower:
                return {'success': True, 'action': action_id, 'error_message': '', 'matched_method': 'string_exact'}
            simplified = re.sub(r'[^a-z0-9]', '', action_str)
            if simplified and simplified in response_simplified:
                return {'success': True, 'action': action_id, 'error_message': '', 'matched_method': 'string_simplified'}
        
        return {
            'success': False,
            'action': None,
            'error_message': f"Cannot parse action from: '{response_clean}'. Expected format: just the action ID number (e.g., '5').",
            'matched_method': 'failed'
        }

    def _build_action_string_map(self, state, legal_actions: List[int]) -> Dict[str, int]:
        """Build mapping from action strings to action IDs"""
        action_map = {}
        for action in legal_actions:
            action_str = state.action_to_string(self._player_id, action).lower()
            action_map[action_str] = action
            if simplified := re.sub(r'[^a-z0-9]', '', action_str):
                action_map[simplified] = action
        return action_map



    def get_conversation(self):
        """Get conversation history (for debugging)"""
        return self._conversation
    
    def get_action_history(self):
        """Get complete action history for all players"""
        return self._action_history

    def get_last_error(self):
        """Get last error string (if any)"""
        return self._last_error

    def get_total_usage(self):
        """Get accumulated usage statistics"""
        return self._total_usage
    
    def get_observation(self):
        """Get final observation string"""
        return self._observation

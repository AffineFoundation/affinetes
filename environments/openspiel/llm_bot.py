"""LLM Bot implementation for OpenSpiel with conversation history support"""

import pyspiel
import numpy as np
import asyncio
import re
import concurrent.futures
import time
from typing import Callable, Awaitable, Tuple, Optional, Dict, List

from base_agent import BaseGameAgent

# Constants
DEFAULT_MAX_PARSING_RETRIES = 3
DEFAULT_MAX_API_RETRIES = 10
API_RETRY_DELAY_SECONDS = 2.0


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
        llm_chat_fn: Callable[[List[Dict]], Awaitable[Tuple[str, Dict]]],
        rng_seed: int,
        agent: BaseGameAgent,
        max_parsing_retries: int = DEFAULT_MAX_PARSING_RETRIES,
        max_api_retries: int = DEFAULT_MAX_API_RETRIES,
    ):
        """
        Initialize LLM Bot with conversation history support

        Args:
            game: pyspiel.Game instance
            player_id: Player ID (0 or 1)
            llm_chat_fn: Async function to call LLM API with messages list
            rng_seed: Random seed for fallback action selection
            agent: BaseGameAgent for game-specific logic (REQUIRED)
            max_parsing_retries: Maximum parsing retry attempts
            max_api_retries: Maximum API call retry attempts
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._llm_chat_fn = llm_chat_fn
        self._rng = np.random.RandomState(rng_seed)
        self._max_parsing_retries = max_parsing_retries
        self._max_api_retries = max_api_retries
        self._agent = agent

        self._messages: List[Dict[str, str]] = []
        self._system_prompt_generated = False
        self._action_history: List[Tuple[int, int]] = []
        self._conversation: List[Dict[str, str]] = []
        self._last_error: Optional[Dict] = None
        self._total_usage = self._init_usage_dict()

    @staticmethod
    def _init_usage_dict() -> Dict[str, int]:
        """Initialize usage statistics dictionary"""
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def restart_at(self, state):
        """Reset to new game"""
        self._messages.clear()
        self._system_prompt_generated = False
        self._action_history.clear()
        self._conversation.clear()
        self._last_error = None
        self._total_usage = self._init_usage_dict()

    def inform_action(self, state, player_id, action):
        """Record other players' actions"""
        self._action_history.append((player_id, action))

    def step(self, state):
        """
        Core method: choose action with conversation history and retry mechanism

        This is called by evaluate_bots during game play.
        """
        legal_actions = state.legal_actions(self._player_id)
        
        # Generate system prompt (first time only)
        if not self._system_prompt_generated:
            system_prompt = self._generate_system_prompt()
            self._messages.append({"role": "system", "content": system_prompt})
            self._conversation.append({"role": "system", "content": system_prompt})
            self._system_prompt_generated = True
        
        # Generate user prompt (current turn)
        user_prompt = self._generate_user_prompt(state)
        self._messages.append({"role": "user", "content": user_prompt})
        
        # Retry loop
        for attempt in range(self._max_parsing_retries + 1):
            # Call LLM API with full message history
            response, usage = self._call_llm_with_api_retry()
            self._messages.append({"role": "assistant", "content": response})
            
            # Parse action
            result = self._parse_action(response, state, legal_actions)
            
            if result['success']:
                # Success: record and return
                action = result['action']
                self._record_successful_turn(user_prompt, response, action, usage, attempt)
                return action
            
            # Parsing failed
            if attempt < self._max_parsing_retries:
                # Add error feedback to messages
                error_msg = self._format_error_feedback(result)
                self._messages.append({"role": "user", "content": error_msg})
                print(f"[Retry {attempt+1}/{self._max_parsing_retries}] {result['error_message']}")
            else:
                # Max retries exceeded
                self._record_parsing_failure(user_prompt, response, result['error_message'], attempt + 1)
                raise ParsingError(
                    f"Failed to parse valid action after {self._max_parsing_retries} retries. "
                    f"Last response: '{response}'. Error: {result['error_message']}"
                )
        
        raise RuntimeError("Should not reach here")

    def _generate_system_prompt(self) -> str:
        """
        Generate system prompt (called once per game)
        
        Uses agent's system prompt
        """
        return self._agent.generate_system_prompt()

    def _generate_user_prompt(self, state) -> str:
        """
        Generate user prompt (called each turn)
        
        Uses agent's user prompt
        """
        return self._agent.generate_user_prompt(
            state=state,
            player_id=self._player_id,
            action_history=self._action_history
        )

    def _format_error_feedback(self, parse_result: dict) -> str:
        """
        Format error feedback for retry (concise, no repetition of rules)
        """
        return (
            f"ERROR: {parse_result['error_message']}\n"
            f"Please respond with ONLY the action ID number."
        )

    def _call_llm_with_api_retry(self) -> Tuple[str, Dict]:
        """Call LLM API with retry mechanism for API failures"""
        for attempt in range(self._max_api_retries):
            try:
                response, usage = self._execute_llm_call()
                self._accumulate_usage(usage)
                return response, usage
            except Exception as e:
                if attempt < self._max_api_retries - 1:
                    print(f"[API Retry {attempt + 1}/{self._max_api_retries}] LLM call failed: {e}, retrying...")
                    time.sleep(API_RETRY_DELAY_SECONDS)
                else:
                    self._handle_api_failure(e)

    def _execute_llm_call(self) -> Tuple[str, Dict]:
        """Execute LLM API call in thread with event loop management"""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._llm_chat_fn(self._messages))
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                return result
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(run_async).result()

    def _accumulate_usage(self, usage: Optional[Dict]):
        """Accumulate token usage statistics"""
        if usage:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._total_usage[key] += usage.get(key, 0)

    def _handle_api_failure(self, error: Exception):
        """Handle final API failure after all retries"""
        import traceback
        error_msg = f"{type(error).__name__}: {str(error)}\n{traceback.format_exc()}"
        print(f"[API Error] LLM call failed after {self._max_api_retries} attempts")
        self._last_error = {
            "type": "api_failure",
            "error": error_msg,
            "attempts": self._max_api_retries,
        }
        raise ParsingError(f"API call failed after {self._max_api_retries} attempts: {error_msg}")

    def _parse_action(self, response: str, state, legal_actions: List[int]) -> Dict:
        """
        Robust action parsing with multiple strategies
        
        Returns dict with keys: success, action, error_message, matched_method
        """
        response_clean = response.strip()
        
        # Strategy 1: Pure number (highest priority)
        if match := re.search(r'^\s*(\d+)\s*$', response_clean):
            action = int(match.group(1))
            return self._create_parse_result(
                action in legal_actions,
                action if action in legal_actions else None,
                '' if action in legal_actions else f"Number {action} not in legal actions: {legal_actions}",
                'pure_number' if action in legal_actions else 'number_invalid'
            )
        
        # Strategy 2: Find legal action ID in text
        for action in legal_actions:
            if re.search(rf'\b{action}\b', response_clean):
                return self._create_parse_result(True, action, '', 'number_in_text')
        
        # Strategy 3: Match action string (exact or simplified)
        action_map = self._build_action_string_map(state, legal_actions)
        response_lower = response_clean.lower()
        response_simplified = re.sub(r'[^a-z0-9]', '', response_lower)
        
        # Try exact match first, then simplified
        for action_str, action_id in action_map.items():
            if action_str in response_lower:
                return self._create_parse_result(True, action_id, '', 'string_exact')
            simplified = re.sub(r'[^a-z0-9]', '', action_str)
            if simplified and simplified in response_simplified:
                return self._create_parse_result(True, action_id, '', 'string_simplified')
        
        return self._create_parse_result(
            False, None,
            f"Cannot parse action from: '{response_clean}'. Expected format: just the action ID number (e.g., '5').",
            'failed'
        )

    def _build_action_string_map(self, state, legal_actions: List[int]) -> Dict[str, int]:
        """Build mapping from action strings to action IDs"""
        action_map = {}
        for action in legal_actions:
            action_str = state.action_to_string(self._player_id, action).lower()
            action_map[action_str] = action
            if simplified := re.sub(r'[^a-z0-9]', '', action_str):
                action_map[simplified] = action
        return action_map

    @staticmethod
    def _create_parse_result(success: bool, action: Optional[int], error_msg: str, method: str) -> Dict:
        """Create standardized parse result dictionary"""
        return {
            'success': success,
            'action': action,
            'error_message': error_msg,
            'matched_method': method
        }

    def _record_successful_turn(
        self, user_prompt: str, response: str, action: int, usage: dict, retry_count: int
    ):
        """Record successful turn in conversation"""
        self._conversation.append({
            "role": "user",
            "content": user_prompt
        })
        self._conversation.append({
            "role": "assistant",
            "content": response
        })
        self._action_history.append((self._player_id, action))

    def _record_parsing_failure(
        self, user_prompt: str, response: str, error: str, total_attempts: int
    ):
        """Record parsing failure"""
        self._last_error = {
            'type': 'parsing_failure',
            'prompt': user_prompt,
            'response': response,
            'error': error,
            'attempts': total_attempts,
        }
        
        self._conversation.append({
            'role': 'system',
            'content': f'[PARSING_FAILURE] {error} (after {total_attempts} attempts)'
        })

    def get_conversation(self):
        """Get conversation history (for debugging)"""
        return self._conversation

    def get_last_error(self):
        """Get last error (if any)"""
        return self._last_error

    def get_total_usage(self):
        """Get accumulated usage statistics"""
        return self._total_usage

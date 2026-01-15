"""OpenSpiel Environment Actor"""

import os
import time
import random
import uuid
import numpy as np
import asyncio
import concurrent.futures
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import mcts
import pyspiel

from llm_bot import LLMBot
from game_config import create_game
from agents import GAME_AGENTS

# Import shared logging utilities
from request_logger import RequestLogger, log_event


class SafeRandomRolloutEvaluator(mcts.Evaluator):
    """
    Safe MCTS evaluator that handles edge cases in Gin Rummy and similar games.
    
    Fixes the "ValueError: 'a' cannot be empty" error that occurs when
    legal_actions() returns an empty list in non-terminal states.
    """
    
    def __init__(self, n_rollouts=1, random_state=None):
        """
        Initialize evaluator
        
        Args:
            n_rollouts: Number of random rollouts per evaluation
            random_state: numpy RandomState for reproducibility
        """
        self._n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()
    
    def evaluate(self, state):
        """
        Evaluate state using random rollouts with safety checks
        
        Args:
            state: OpenSpiel state to evaluate
            
        Returns:
            List of returns for each player
        """
        # If terminal state, return actual returns
        if state.is_terminal():
            return state.returns()
        
        # Safety check: if no legal actions in non-terminal state
        legal_actions = state.legal_actions()
        if not legal_actions:
            # This shouldn't happen in well-formed games, but Gin Rummy has edge cases
            # Return current returns as approximation
            return state.returns()
        
        # Perform n random rollouts
        total_returns = np.zeros(state.num_players())
        
        for _ in range(self._n_rollouts):
            working_state = state.clone()
            
            # Rollout until terminal
            while not working_state.is_terminal():
                legal_actions = working_state.legal_actions()
                
                # Safety check during rollout
                if not legal_actions:
                    # Edge case: non-terminal state with no legal actions
                    # Break and use current returns
                    break
                
                # Choose random action
                action = self._random_state.choice(legal_actions)
                working_state.apply_action(action)
            
            # Accumulate returns
            total_returns += working_state.returns()
        
        # Return average returns across rollouts
        return total_returns / self._n_rollouts
    
    def prior(self, state):
        """
        Return prior policy (uniform distribution over legal actions)
        
        Args:
            state: OpenSpiel state
            
        Returns:
            List of (action, probability) tuples
        """
        legal_actions = state.legal_actions()
        
        # Safety check
        if not legal_actions:
            return []
        
        # Uniform prior
        prob = 1.0 / len(legal_actions)
        return [(action, prob) for action in legal_actions]


class TimedMCTSBot(pyspiel.Bot):
    """Wrapper for MCTS bot that tracks computation time"""
    
    def __init__(self, mcts_bot):
        pyspiel.Bot.__init__(self)
        self._mcts_bot = mcts_bot
        self.total_mcts_time = 0.0
        self.mcts_call_count = 0
    
    def restart_at(self, state):
        self._mcts_bot.restart_at(state)
        self.total_mcts_time = 0.0
        self.mcts_call_count = 0
    
    def inform_action(self, state, player_id, action):
        self._mcts_bot.inform_action(state, player_id, action)
    
    def step(self, state):
        start_time = time.time()
        action = self._mcts_bot.step(state)
        elapsed = time.time() - start_time
        self.total_mcts_time += elapsed
        self.mcts_call_count += 1
        return action
    
    def get_timing_stats(self):
        return {
            'total_mcts_time': self.total_mcts_time,
            'mcts_call_count': self.mcts_call_count,
            'avg_mcts_time_per_step': self.total_mcts_time / self.mcts_call_count if self.mcts_call_count > 0 else 0.0
        }


class Actor:
    """OpenSpiel evaluation wrapper"""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)

    def __init__(self, api_key: str = None):
        """
        Initialize Actor with API key

        Args:
            api_key: API key for LLM service. If not provided, uses CHUTES_API_KEY env var
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        
        # OpenEnv state - for step-based interaction
        self._game = None
        self._game_state = None
        self._game_config = None
        self._agent = None
        self._opponent_bots = {}  # player_id -> bot
        self._llm_player_id = None
        self._episode_seed = None
        self._episode_done = True
        self._episode_start_time = None
        self._conversation = []  # Track conversation history
        self._action_history = []  # Track all actions
        self._rng = None  # Random state for game

    async def evaluate(
        self,
        task_id: int = None,
        seed: int = None,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 1800,
        temperature: float = None,
        api_key: str = None,
        opponent: str = "mcts",
    ):
        """
        Run single game evaluation

        Args:
            task_id: Task identifier (12-digit format: GGGGCCCCCCCC)
            seed: Random seed for reproducibility
            model: LLM model name
            base_url: LLM API base URL
            timeout: Overall task timeout in seconds (default 1800s = 30min)
            temperature: LLM temperature (None = use model default)
            api_key: Override API key
            opponent: Opponent type ("random" or "mcts")
        """
        if task_id is None:
            task_id = random.randint(0, 10**11 - 1)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        current_api_key = api_key or self.api_key
        start_time = time.time()

        return await asyncio.wait_for(
            self._run_evaluation(
                task_id,
                seed,
                model,
                base_url,
                temperature,
                current_api_key,
                opponent,
                start_time,
                timeout,
            ),
            timeout=timeout,
        )

    async def _run_evaluation(
        self,
        task_id,
        seed,
        model,
        base_url,
        temperature,
        current_api_key,
        opponent,
        start_time,
        task_timeout,
    ):
        """Internal method to run evaluation with unified error handling"""
        llm_player_id = seed % 2
        game_name = "unknown"
        llm_bot = None
        mcts_bots = []  # Track MCTS bots for timing stats
        logger = None

        # Set internal timeout to be 20 seconds earlier than task timeout
        # This allows us to gracefully finish and return partial results
        internal_timeout = max(task_timeout - 20, task_timeout * 0.9)

        try:
            game, game_config = create_game(task_id)
            game_name = game_config["game_name"]

            # Setup logging after game_name is determined
            logger = RequestLogger(
                task_id=task_id,
                task_type=game_name,
                seed=seed,
                model=model,
                base_url=base_url,
                opponent=opponent
            )
            logger.__enter__()
            log_event("game_created", game_name=game_name)
            num_players = game.num_players()
            llm_player_id = llm_player_id % num_players

            # Get agent for this game
            agent_class = GAME_AGENTS.get(game_name)
            if not agent_class:
                raise ValueError(f"No agent found for game: {game_name}")
            
            agent = agent_class()

            llm_bot = LLMBot(
                game=game,
                player_id=llm_player_id,
                base_url=base_url,
                api_key=current_api_key,
                model=model,
                temperature=temperature,
                rng_seed=seed + 1,
                agent=agent,
                seed=seed,
                executor=self.executor,
            )

            # Create bots for all players
            bots = []
            for player_id in range(num_players):
                if player_id == llm_player_id:
                    bots.append(llm_bot)
                else:
                    opponent_bot = self._create_opponent_bot(
                        opponent, player_id, seed + 2 + player_id, game, agent
                    )
                    # Track TimedMCTSBot instances
                    if isinstance(opponent_bot, TimedMCTSBot):
                        mcts_bots.append(opponent_bot)
                    bots.append(opponent_bot)

            loop = asyncio.get_event_loop()
            log_event("game_start", num_players=num_players, llm_player_id=llm_player_id)

            # Run game evaluation with internal timeout (20s buffer before task timeout)
            try:
                returns = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        evaluate_bots.evaluate_bots,
                        game.new_initial_state(),
                        bots,
                        np.random.RandomState(seed),
                    ),
                    timeout=internal_timeout
                )
                log_event("game_complete", returns=str(returns))
            except asyncio.TimeoutError:
                # Internal timeout - game didn't complete in time
                log_event("game_timeout", level='warning', timeout_seconds=internal_timeout)
                elapsed = time.time() - start_time
                result = self._build_result(
                    game_name=game_name,
                    llm_player_id=llm_player_id,
                    task_id=task_id,
                    seed=seed,
                    opponent=opponent,
                    start_time=start_time,
                    error=f"Game incomplete: timeout after {elapsed:.1f}s (limit: {task_timeout}s)",
                    llm_bot=llm_bot,
                    mcts_bots=mcts_bots,
                )
                if logger:
                    logger.__exit__(None, None, None)
                return result

            # Game completed successfully
            llm_return = returns[llm_player_id]
            score = self._compute_score(returns, llm_player_id, game)
            log_event("request_complete", score=score, llm_return=llm_return)

            result = self._build_result(
                game_name=game_name,
                llm_player_id=llm_player_id,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
                score=score,
                llm_return=llm_return,
                all_returns=returns,
                error=llm_bot.get_last_error() if llm_bot else None,
                llm_bot=llm_bot,
                mcts_bots=mcts_bots,
            )
            if logger:
                logger.__exit__(None, None, None)
            return result

        except asyncio.TimeoutError:
            # Task timeout
            result = self._build_result(
                game_name=game_name,
                llm_player_id=llm_player_id,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
                error=f"Task timeout exceeded ({task_timeout}s)",
                llm_bot=llm_bot,
                mcts_bots=mcts_bots,
            )
            if logger:
                logger.__exit__(None, None, None)
            return result

        except Exception as e:
            import traceback
            from llm_bot import ParsingError, APIError

            # ParsingError: treat as valid sample with 0 score (no error field)
            if isinstance(e, ParsingError):
                result = self._build_result(
                    game_name=game_name,
                    llm_player_id=llm_player_id,
                    task_id=task_id,
                    seed=seed,
                    opponent=opponent,
                    start_time=start_time,
                    score=0.0,
                    error=None,  # No error - valid sample
                    llm_bot=llm_bot,
                    mcts_bots=mcts_bots,
                )
                if logger:
                    logger.__exit__(None, None, None)
                return result

            # APIError or other exceptions: record as error
            if isinstance(e, APIError):
                error_msg = llm_bot.get_last_error() if llm_bot and llm_bot.get_last_error() else str(e)
            elif llm_bot and llm_bot.get_last_error():
                error_msg = llm_bot.get_last_error()
            else:
                error_msg = f"[{type(e).__name__}] {str(e)}\n{traceback.format_exc()}"

            result = self._build_result(
                game_name=game_name,
                llm_player_id=llm_player_id,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
                error=error_msg,
                llm_bot=llm_bot,
                mcts_bots=mcts_bots,
            )
            if logger:
                logger.__exit__(None, None, None)
            return result

    def _compute_score(self, returns, llm_player_idx, game):
        """
        Compute normalized score [0.0, 1.0] from OpenSpiel returns.
        
        This method respects the game type (zero-sum, general-sum, etc.)
        to properly convert raw returns into a meaningful score.
        
        Args:
            returns: Terminal returns from state.returns()
            llm_player_idx: Index of LLM player
            game: OpenSpiel game object
        
        Returns:
            Normalized score in [0.0, 1.0]
        """
        num_players = len(returns)
        llm_return = returns[llm_player_idx]
        game_type = game.get_type()
        
        # Zero-sum games (e.g., Chess, Poker): returns are in game's utility range
        if game_type.utility == pyspiel.GameType.Utility.ZERO_SUM:
            # Normalize from [min_utility, max_utility] to [0, 1]
            # Example: Chess has [-1, 1] → Loss:-1→0.0, Draw:0→0.5, Win:1→1.0
            min_utility = game.min_utility()
            max_utility = game.max_utility()
            if max_utility > min_utility:
                score = (llm_return - min_utility) / (max_utility - min_utility)
            else:
                score = 0
            return float(score)
        
        # Multi-player games (3-4 players): use ranking-based scoring
        if num_players > 2:
            # Rank players by returns (higher return = better performance)
            sorted_returns = sorted(returns, reverse=True)
            llm_rank = sorted_returns.index(llm_return)
            
            # Convert rank to score: 1st→1.0, 2nd→0.67, 3rd→0.33, 4th→0.0
            # This preserves discrimination between different ranks
            score = 1.0 - (llm_rank / (num_players - 1))
            return float(score)
        
        # 2-player non-zero-sum games: compare relative performance
        if num_players == 2:
            opponent_return = returns[1 - llm_player_idx]
            
            # Determine winner by comparing returns (higher is better)
            if llm_return > opponent_return:
                return 1.0
            elif llm_return < opponent_return:
                return 0.0
            else:
                return 0.5  # Tie
        
        # Fallback: normalize by game's utility range (for unusual game types)
        min_utility = game.min_utility()
        max_utility = game.max_utility()
        if max_utility > min_utility:
            score = (llm_return - min_utility) / (max_utility - min_utility)
        else:
            score = 0.5
        return float(score)

    def _create_opponent_bot(self, opponent, player_id, seed, game, agent):
        """Create opponent bot based on type and game dynamics"""
        game_type = game.get_type()
        # For simultaneous move games, MCTS doesn't work - fallback to random
        if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            return uniform_random.UniformRandomBot(
                player_id=player_id, rng=np.random.RandomState(seed + 2)
            )
        
        # For sequential games, use requested opponent type
        if opponent == "random":
            return uniform_random.UniformRandomBot(
                player_id=player_id, rng=np.random.RandomState(seed + 2)
            )
        elif opponent == "mcts":
            # Get MCTS config from agent
            mcts_config = agent.get_mcts_config()
            
            # If agent returns None, game doesn't need MCTS (e.g., single-player)
            if mcts_config is None:
                return uniform_random.UniformRandomBot(
                    player_id=player_id, rng=np.random.RandomState(seed + 2)
                )
            
            max_simulations, n_rollouts = mcts_config
            
            # Create a safe evaluator that handles edge cases
            evaluator = SafeRandomRolloutEvaluator(
                n_rollouts=n_rollouts, random_state=np.random.RandomState(seed + 3)
            )
            mcts_bot = mcts.MCTSBot(
                game=game,
                uct_c=1.414,
                max_simulations=max_simulations,
                evaluator=evaluator,
                random_state=np.random.RandomState(seed + 4),
            )
            # Wrap with timing tracker
            return TimedMCTSBot(mcts_bot)
        else:
            raise ValueError(f"Unknown opponent type: {opponent}")

    def _build_result(
        self,
        game_name,
        llm_player_id,
        task_id,
        seed,
        opponent,
        start_time,
        score=0.0,
        llm_return=None,
        all_returns=None,
        error=None,
        llm_bot=None,
        mcts_bots=None,
    ):
        """Build result dictionary with automatic data extraction
        
        Args:
            game_name: Name of the game
            llm_player_id: LLM player index
            task_id: Task identifier
            seed: Random seed
            opponent: Opponent type
            start_time: Evaluation start time
            score: Normalized score (default: 0.0)
            llm_return: Raw return value (default: None)
            all_returns: All players' returns (default: None)
            error: Error message if any (default: None)
            llm_bot: LLMBot instance to extract conversation/usage (default: None)
            mcts_bots: List of TimedMCTSBot instances for timing stats (default: None)
        """
        # Extract conversation, action_history, final_state, and usage from llm_bot
        conversation = []
        action_history = []
        observation = None
        usage = None
        if llm_bot is not None:
            try:
                conversation = llm_bot.get_conversation()
                action_history = llm_bot.get_action_history()
                observation = llm_bot.get_observation()
                usage = llm_bot.get_total_usage()
            except:
                pass
        
        # Collect MCTS timing stats
        mcts_stats = None
        if mcts_bots:
            total_time = sum(bot.total_mcts_time for bot in mcts_bots)
            total_calls = sum(bot.mcts_call_count for bot in mcts_bots)
            mcts_stats = {
                'total_mcts_time': total_time,
                'total_mcts_calls': total_calls,
                'avg_mcts_time_per_call': total_time / total_calls if total_calls > 0 else 0.0,
                'num_mcts_bots': len(mcts_bots)
            }
        
        # Build result
        result = {
            "task_name": f"openspiel:{game_name}",
            "score": score,
            "success": score > 0.5,
            "time_taken": time.time() - start_time,
            "extra": {
                "conversation": conversation,
                "action_history": action_history,
                "observation": observation,
                "task_type": game_name,
                "game_name": game_name,
                "task_id": task_id,
                "seed": seed,
                "opponent_type": opponent,
                "llm_player_id": llm_player_id,
                "final_return": llm_return,
                "all_returns": all_returns,
                "usage": usage
                or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        }

        # Add MCTS timing stats if available
        if mcts_stats:
            result["extra"]["mcts_timing"] = mcts_stats

        # Add error to top-level (consistent with other environments)
        if error:
            result["error"] = str(error)

        return result

    # =========================================================================
    # OpenEnv Protocol Methods
    # =========================================================================
    
    def reset(
        self,
        task_id: int = None,
        seed: int = None,
        opponent: str = "random",
    ) -> dict:
        """
        OpenEnv reset: Initialize a new game episode and return initial observation.
        
        Args:
            task_id: Task identifier (12-digit format: GGGGCCCCCCCC).
                     If not provided, a random task is selected.
            seed: Random seed for reproducibility.
                  If not provided, a random seed is generated.
            opponent: Opponent type ("random" or "mcts")
        
        Returns:
            dict with:
                - observation: The game state prompt (text for LLM)
                - reward: 0.0 (no reward at reset)
                - done: False (episode just started)
                - truncated: False
                - info: Additional metadata including legal_actions
        """
        # Generate defaults if not provided
        if task_id is None:
            task_id = random.randint(0, 10**11 - 1)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        self._episode_seed = seed
        self._episode_start_time = time.time()
        self._episode_done = False
        self._conversation = []
        self._action_history = []
        
        # Create game from task_id
        self._game, self._game_config = create_game(task_id)
        game_name = self._game_config["game_name"]
        num_players = self._game.num_players()
        
        # Determine LLM player ID from seed
        self._llm_player_id = seed % num_players
        
        # Get agent for this game
        agent_class = GAME_AGENTS.get(game_name)
        if not agent_class:
            raise ValueError(f"No agent found for game: {game_name}")
        self._agent = agent_class()
        
        # Create opponent bots for all non-LLM players
        self._opponent_bots = {}
        for player_id in range(num_players):
            if player_id != self._llm_player_id:
                self._opponent_bots[player_id] = self._create_opponent_bot(
                    opponent, player_id, seed + 2 + player_id, self._game, self._agent
                )
        
        # Initialize random state for game
        self._rng = np.random.RandomState(seed)
        
        # Create initial game state
        self._game_state = self._game.new_initial_state()
        
        # Generate system prompt and add to conversation
        system_prompt = self._agent.generate_system_prompt()
        self._conversation.append({"role": "system", "content": system_prompt})
        
        # Advance game to first LLM turn (handle chance nodes and opponent turns)
        self._advance_to_llm_turn()
        
        # Check if game ended during advancement (e.g., game with no LLM moves)
        if self._game_state.is_terminal():
            self._episode_done = True
            returns = self._game_state.returns()
            score = self._compute_score(returns, self._llm_player_id, self._game)
            return {
                "observation": "",
                "reward": score,
                "done": True,
                "truncated": False,
                "info": self._build_info(score=score, returns=returns)
            }
        
        # Generate observation for LLM
        observation = self._generate_observation()
        
        return {
            "observation": observation,
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "info": self._build_info()
        }
    
    def step(self, action: str) -> dict:
        """
        OpenEnv step: Process an action (LLM response) and return result.
        
        Args:
            action: The action to take (can be action ID or action string)
        
        Returns:
            dict with:
                - observation: Next state prompt (empty if done)
                - reward: Game reward (computed at terminal state)
                - done: Whether the episode has ended
                - truncated: False (we don't truncate games)
                - info: Additional metadata
        """
        if self._game_state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        
        if self._episode_done:
            raise RuntimeError("Episode already done. Call reset() to start a new episode.")
        
        if self._game_state.is_terminal():
            self._episode_done = True
            returns = self._game_state.returns()
            score = self._compute_score(returns, self._llm_player_id, self._game)
            return {
                "observation": "",
                "reward": score,
                "done": True,
                "truncated": False,
                "info": self._build_info(score=score, returns=returns)
            }
        
        # Parse the action
        legal_actions = self._game_state.legal_actions(self._llm_player_id)
        parsed_action = self._parse_action_string(action, legal_actions)
        
        if parsed_action is None:
            # Invalid action - return error but don't end episode
            # Let the training loop retry or handle the error
            return {
                "observation": self._generate_observation(),
                "reward": 0.0,
                "done": False,
                "truncated": False,
                "info": self._build_info(
                    error=f"Invalid action: '{action}'. Legal actions: {legal_actions}"
                )
            }
        
        # Record LLM's action
        self._record_action(self._llm_player_id, parsed_action)
        
        # Add assistant response to conversation
        self._conversation.append({"role": "assistant", "content": action})
        
        # Apply the action
        self._game_state.apply_action(parsed_action)
        
        # Advance to next LLM turn (handle chance nodes and opponent turns)
        self._advance_to_llm_turn()
        
        # Check if game ended
        if self._game_state.is_terminal():
            self._episode_done = True
            returns = self._game_state.returns()
            score = self._compute_score(returns, self._llm_player_id, self._game)
            return {
                "observation": "",
                "reward": score,
                "done": True,
                "truncated": False,
                "info": self._build_info(score=score, returns=returns)
            }
        
        # Generate next observation for LLM
        observation = self._generate_observation()
        
        return {
            "observation": observation,
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "info": self._build_info()
        }
    
    def state(self) -> dict:
        """
        OpenEnv state: Return current observation without taking any action.
        
        Returns:
            dict with:
                - observation: Current game state prompt (or empty if no active episode)
                - reward: 0.0
                - done: Current done status
                - truncated: False
                - info: Current episode metadata
        """
        if self._game_state is None:
            return {
                "observation": "",
                "reward": 0.0,
                "done": True,
                "truncated": False,
                "info": {"error": "No active episode. Call reset() first."}
            }
        
        if self._episode_done or self._game_state.is_terminal():
            returns = self._game_state.returns() if self._game_state.is_terminal() else None
            score = self._compute_score(returns, self._llm_player_id, self._game) if returns else 0.0
            return {
                "observation": "",
                "reward": score,
                "done": True,
                "truncated": False,
                "info": self._build_info(score=score, returns=returns)
            }
        
        return {
            "observation": self._generate_observation(),
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "info": self._build_info()
        }
    
    def _advance_to_llm_turn(self):
        """
        Advance game state until it's the LLM's turn or game is terminal.
        
        Handles:
        - Chance nodes (random events)
        - Opponent bot turns
        """
        while not self._game_state.is_terminal():
            current_player = self._game_state.current_player()
            
            # Handle chance nodes
            if current_player == pyspiel.PlayerId.CHANCE:
                outcomes = self._game_state.chance_outcomes()
                action_probs = [(a, p) for a, p in outcomes]
                actions = [a for a, _ in action_probs]
                probs = [p for _, p in action_probs]
                action = self._rng.choice(actions, p=probs)
                self._game_state.apply_action(action)
                continue
            
            # Handle simultaneous games - need LLM to act
            if current_player == pyspiel.PlayerId.SIMULTANEOUS:
                # For simultaneous games, we need all players to act at once
                # This is a special case - return to let LLM provide action
                break
            
            # If it's LLM's turn, stop advancing
            if current_player == self._llm_player_id:
                break
            
            # Opponent's turn - let bot play
            if current_player in self._opponent_bots:
                bot = self._opponent_bots[current_player]
                action = bot.step(self._game_state)
                self._record_action(current_player, action)
                
                # Inform all bots about the action
                for pid, other_bot in self._opponent_bots.items():
                    if hasattr(other_bot, 'inform_action'):
                        other_bot.inform_action(self._game_state, current_player, action)
                
                self._game_state.apply_action(action)
            else:
                # Unknown player - shouldn't happen
                break
    
    def _generate_observation(self) -> str:
        """
        Generate observation text for LLM (the prompt).
        
        Returns the same format as the LLMBot would generate.
        """
        if self._game_state.is_terminal():
            return ""
        
        # Get legal actions for the LLM player
        current_player = self._game_state.current_player()
        
        # Handle simultaneous games
        if current_player == pyspiel.PlayerId.SIMULTANEOUS:
            legal_actions = self._game_state.legal_actions(self._llm_player_id)
        else:
            legal_actions = self._game_state.legal_actions(self._llm_player_id)
        
        # Generate user prompt using the agent
        user_prompt = self._agent.generate_user_prompt(
            state=self._game_state,
            player_id=self._llm_player_id,
            legal_actions=legal_actions
        )
        
        # Add to conversation history
        self._conversation.append({"role": "user", "content": user_prompt})
        
        return user_prompt
    
    def _parse_action_string(self, action_str: str, legal_actions: list) -> int:
        """
        Parse action string to action ID.
        
        Supports:
        - Pure number: "42"
        - Number in text: "I choose 42"
        - Action string match: "a3" for chess
        """
        action_str = action_str.strip()
        
        # Strategy 1: Pure number
        try:
            action = int(action_str)
            if action in legal_actions:
                return action
        except ValueError:
            pass
        
        # Strategy 2: Find number in text
        import re
        for action in legal_actions:
            if re.search(rf'\b{action}\b', action_str):
                return action
        
        # Strategy 3: Match action string
        action_str_lower = action_str.lower()
        for action in legal_actions:
            try:
                game_action_str = self._game_state.action_to_string(self._llm_player_id, action).lower()
                if game_action_str in action_str_lower or action_str_lower in game_action_str:
                    return action
            except:
                pass
        
        return None
    
    def _record_action(self, player_id: int, action: int):
        """Record an action to the action history."""
        try:
            action_str = self._game_state.action_to_string(player_id, action)
        except:
            action_str = str(action)
        
        self._action_history.append({
            "player_id": int(player_id),
            "action": int(action),
            "action_str": action_str,
            "is_llm": bool(player_id == self._llm_player_id)
        })
    
    def _build_info(self, score: float = None, returns: list = None, error: str = None) -> dict:
        """Build info dictionary for OpenEnv response."""
        game_name = self._game_config["game_name"] if self._game_config else "unknown"
        
        # Get legal actions if game is not terminal
        legal_actions = []
        legal_actions_str = []
        if self._game_state and not self._game_state.is_terminal():
            current_player = self._game_state.current_player()
            if current_player == pyspiel.PlayerId.SIMULTANEOUS:
                legal_actions = list(self._game_state.legal_actions(self._llm_player_id))
            elif current_player == self._llm_player_id:
                legal_actions = list(self._game_state.legal_actions(self._llm_player_id))
            
            # Convert to strings for readability
            for action in legal_actions:
                try:
                    action_str = self._game_state.action_to_string(self._llm_player_id, action)
                    legal_actions_str.append(f"{action} -> {action_str}")
                except:
                    legal_actions_str.append(str(action))
        
        info = {
            "game_name": game_name,
            "task_id": self._game_config.get("task_id") if self._game_config else None,
            "seed": self._episode_seed,
            "llm_player_id": self._llm_player_id,
            "legal_actions": legal_actions,
            "legal_actions_str": legal_actions_str,
            "action_history": self._action_history,
            "conversation": self._conversation,
            "time_elapsed": time.time() - self._episode_start_time if self._episode_start_time else 0,
        }
        
        if score is not None:
            info["score"] = score
            info["success"] = score > 0.5
        
        if returns is not None:
            info["returns"] = [float(r) for r in returns]
            info["llm_return"] = float(returns[self._llm_player_id]) if self._llm_player_id < len(returns) else None
        
        if error is not None:
            info["error"] = error
        
        return info

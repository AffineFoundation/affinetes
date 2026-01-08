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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_bot import LLMBot
from local_llm_bot import LocalLLMBot
from game_config import create_game
from agents import GAME_AGENTS


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

    async def evaluate(
        self,
        task_id: int = None,
        seed: int = None,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 1800,
        temperature: float = 0.7,
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
            temperature: LLM temperature
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
        
        # Set internal timeout to be 20 seconds earlier than task timeout
        # This allows us to gracefully finish and return partial results
        internal_timeout = max(task_timeout - 20, task_timeout * 0.9)
        
        try:
            game, game_config = create_game(task_id)
            game_name = game_config["game_name"]
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
            except asyncio.TimeoutError:
                # Internal timeout - game didn't complete in time
                elapsed = time.time() - start_time
                return self._build_result(
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

            # Game completed successfully
            llm_return = returns[llm_player_id]
            score = self._compute_score(returns, llm_player_id, game)
            
            return self._build_result(
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

        except asyncio.TimeoutError:
            # Task timeout
            return self._build_result(
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

        except Exception as e:
            import traceback
            from llm_bot import ParsingError, APIError

            # ParsingError: treat as valid sample with 0 score (no error field)
            if isinstance(e, ParsingError):
                return self._build_result(
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
            
            # APIError or other exceptions: record as error
            if isinstance(e, APIError):
                error_msg = llm_bot.get_last_error() if llm_bot and llm_bot.get_last_error() else str(e)
            elif llm_bot and llm_bot.get_last_error():
                error_msg = llm_bot.get_last_error()
            else:
                error_msg = f"[{type(e).__name__}] {str(e)}\n{traceback.format_exc()}"

            return self._build_result(
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

    async def local_evaluate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task_id: int = None,
        seed: int = None,
        timeout: int = 1800,
        temperature: float = 0.7,
        opponent: str = "mcts",
    ):
        """
        Run single game evaluation using local model

        Args:
            model: Loaded AutoModelForCausalLM instance
            tokenizer: Loaded AutoTokenizer instance
            task_id: Task identifier (12-digit format: GGGGCCCCCCCC)
            seed: Random seed for reproducibility
            timeout: Overall task timeout in seconds (default 1800s = 30min)
            opponent: Opponent type ("random" or "mcts")
        """
        if task_id is None:
            task_id = random.randint(0, 10**11 - 1)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        start_time = time.time()

        return await asyncio.wait_for(
            self._run_local_evaluation(
                task_id,
                seed,
                model,
                tokenizer,
                opponent,
                start_time,
                timeout,
                temperature,
            ),
            timeout=timeout,
        )

    async def _run_local_evaluation(
        self,
        task_id,
        seed,
        model,
        tokenizer,
        opponent,
        start_time,
        task_timeout,
        temperature,
    ):
        """Internal method to run local evaluation with unified error handling"""
        from local_llm_bot import ParsingError

        llm_player_id = seed % 2
        game_name = "unknown"
        llm_bot = None
        mcts_bots = []

        internal_timeout = max(task_timeout - 20, task_timeout * 0.9)

        try:
            game, game_config = create_game(task_id)
            game_name = game_config["game_name"]
            num_players = game.num_players()
            llm_player_id = llm_player_id % num_players

            agent_class = GAME_AGENTS.get(game_name)
            if not agent_class:
                raise ValueError(f"No agent found for game: {game_name}")

            agent = agent_class()

            llm_bot = LocalLLMBot(
                game=game,
                player_id=llm_player_id,
                model=model,
                temperature=temperature,
                tokenizer=tokenizer,
                rng_seed=seed + 1,
                agent=agent,
                seed=seed,
            )

            bots = []
            for player_id in range(num_players):
                if player_id == llm_player_id:
                    bots.append(llm_bot)
                else:
                    opponent_bot = self._create_opponent_bot(
                        opponent, player_id, seed + 2 + player_id, game, agent
                    )
                    if isinstance(opponent_bot, TimedMCTSBot):
                        mcts_bots.append(opponent_bot)
                    bots.append(opponent_bot)

            loop = asyncio.get_event_loop()

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
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                return self._build_local_result(
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

            llm_return = returns[llm_player_id]
            score = self._compute_score(returns, llm_player_id, game)

            return self._build_local_result(
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

        except asyncio.TimeoutError:
            return self._build_local_result(
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

        except Exception as e:
            import traceback

            if isinstance(e, ParsingError):
                return self._build_local_result(
                    game_name=game_name,
                    llm_player_id=llm_player_id,
                    task_id=task_id,
                    seed=seed,
                    opponent=opponent,
                    start_time=start_time,
                    score=0.0,
                    error=None,
                    llm_bot=llm_bot,
                    mcts_bots=mcts_bots,
                )

            if llm_bot and llm_bot.get_last_error():
                error_msg = llm_bot.get_last_error()
            else:
                error_msg = f"[{type(e).__name__}] {str(e)}\n{traceback.format_exc()}"

            return self._build_local_result(
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

    def _build_local_result(
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
        """Build result dictionary for local evaluation"""
        conversation = []
        action_history = []
        observation = None
        if llm_bot is not None:
            try:
                conversation = llm_bot.get_conversation()
                action_history = llm_bot.get_action_history()
                observation = llm_bot.get_observation()
            except:
                pass

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

        result = {
            "task_name": f"openspiel:{game_name}",
            "score": score,
            "success": score > 0.5,
            "time_taken": time.time() - start_time,
            "extra": {
                "conversation": conversation,
                "action_history": action_history,
                "observation": observation,
                "game_name": game_name,
                "task_id": task_id,
                "seed": seed,
                "opponent_type": opponent,
                "llm_player_id": llm_player_id,
                "final_return": llm_return,
                "all_returns": all_returns,
            },
        }

        if mcts_stats:
            result["extra"]["mcts_timing"] = mcts_stats

        if error:
            result["error"] = str(error)

        return result


def play_game(
    bot,
    task_id: int = None,
    seed: int = None,
    opponent: str = "mcts",
    player_id: int = 0,
) -> dict:
    """
    Play a game with any pyspiel.Bot (synchronous version).

    This function is designed for:
    - Human play via CLI (with HumanBot)
    - Trajectory generation for LLM training (with AlgorithmBot)

    Args:
        bot: Your pyspiel.Bot instance (HumanBot, AlgorithmBot, etc.)
        task_id: Task identifier for game configuration. If None, random.
        seed: Random seed for reproducibility. If None, random.
        opponent: Opponent type ("random" or "mcts")
        player_id: Which position your bot plays (0, 1, ...)

    Returns:
        dict with:
            - conversation: List of messages in LLM format (if bot supports it)
            - action_history: List of all actions taken
            - score: Normalized score [0.0, 1.0] for your bot
            - returns: Raw returns for all players
            - game_name: Name of the game
            - task_id, seed: For reproducibility
    """
    import time

    if task_id is None:
        task_id = random.randint(0, 10**11 - 1)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    start_time = time.time()

    try:
        # Create game from task_id
        game, game_config = create_game(task_id)
        game_name = game_config["game_name"]
        num_players = game.num_players()
        player_id = player_id % num_players

        # Get agent for this game
        agent_class = GAME_AGENTS.get(game_name)
        if not agent_class:
            raise ValueError(f"No agent found for game: {game_name}")
        agent = agent_class()

        # Create bots for all players
        bots = []
        for pid in range(num_players):
            if pid == player_id:
                bots.append(bot)
            else:
                # Create opponent using same logic as Actor
                opponent_bot = _create_opponent_bot(
                    opponent, pid, seed + 2 + pid, game, agent
                )
                bots.append(opponent_bot)

        # Run game
        returns = evaluate_bots.evaluate_bots(
            game.new_initial_state(),
            bots,
            np.random.RandomState(seed),
        )

        # Compute score
        score = _compute_score(returns, player_id, game)

        # Extract data from bot if available
        conversation = []
        action_history = []
        observation = None

        if hasattr(bot, 'get_conversation'):
            conversation = bot.get_conversation()
        if hasattr(bot, 'get_action_history'):
            action_history = bot.get_action_history()
        if hasattr(bot, 'get_observation'):
            observation = bot.get_observation()

        return {
            "task_name": f"openspiel:{game_name}",
            "score": score,
            "success": score > 0.5,
            "time_taken": time.time() - start_time,
            "conversation": conversation,
            "action_history": action_history,
            "observation": observation,
            "game_name": game_name,
            "task_id": task_id,
            "seed": seed,
            "opponent_type": opponent,
            "player_id": player_id,
            "returns": list(returns),
            "player_return": returns[player_id],
        }

    except Exception as e:
        import traceback
        return {
            "task_name": f"openspiel:unknown",
            "score": 0.0,
            "success": False,
            "time_taken": time.time() - start_time,
            "error": f"[{type(e).__name__}] {str(e)}\n{traceback.format_exc()}",
            "task_id": task_id,
            "seed": seed,
        }


def _create_opponent_bot(opponent: str, player_id: int, seed: int, game, agent):
    """Create opponent bot based on type (standalone version for play_game)"""
    game_type = game.get_type()

    # For simultaneous move games, MCTS doesn't work - fallback to random
    if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        return uniform_random.UniformRandomBot(
            player_id=player_id, rng=np.random.RandomState(seed)
        )

    if opponent == "random":
        return uniform_random.UniformRandomBot(
            player_id=player_id, rng=np.random.RandomState(seed)
        )
    elif opponent == "mcts":
        mcts_config = agent.get_mcts_config()

        if mcts_config is None:
            return uniform_random.UniformRandomBot(
                player_id=player_id, rng=np.random.RandomState(seed)
            )

        max_simulations, n_rollouts = mcts_config

        evaluator = SafeRandomRolloutEvaluator(
            n_rollouts=n_rollouts, random_state=np.random.RandomState(seed + 1)
        )
        return mcts.MCTSBot(
            game=game,
            uct_c=1.414,
            max_simulations=max_simulations,
            evaluator=evaluator,
            random_state=np.random.RandomState(seed + 2),
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent}")


def _compute_score(returns, player_idx: int, game) -> float:
    """Compute normalized score [0.0, 1.0] (standalone version for play_game)"""
    num_players = len(returns)
    player_return = returns[player_idx]
    game_type = game.get_type()

    # Zero-sum games
    if game_type.utility == pyspiel.GameType.Utility.ZERO_SUM:
        min_utility = game.min_utility()
        max_utility = game.max_utility()
        if max_utility > min_utility:
            return float((player_return - min_utility) / (max_utility - min_utility))
        return 0.5

    # Multi-player games: ranking-based
    if num_players > 2:
        sorted_returns = sorted(returns, reverse=True)
        rank = sorted_returns.index(player_return)
        return float(1.0 - (rank / (num_players - 1)))

    # 2-player non-zero-sum
    if num_players == 2:
        opponent_return = returns[1 - player_idx]
        if player_return > opponent_return:
            return 1.0
        elif player_return < opponent_return:
            return 0.0
        return 0.5

    # Fallback
    min_utility = game.min_utility()
    max_utility = game.max_utility()
    if max_utility > min_utility:
        return float((player_return - min_utility) / (max_utility - min_utility))
    return 0.5


def _play_game_worker(args: dict) -> dict:
    """
    Worker function for parallel game execution.

    Creates bot internally to avoid pickling issues with pyspiel objects.

    Args:
        args: dict with task_id, seed, algorithm, opponent, player_id, mcts_simulations

    Returns:
        Game result dict
    """
    task_id = args["task_id"]
    seed = args.get("seed") or random.randint(0, 2**32 - 1)
    algorithm = args.get("algorithm", "mcts")
    opponent = args.get("opponent", "mcts")
    player_id = args.get("player_id", 0)
    mcts_simulations = args.get("mcts_simulations")

    # Import here to avoid issues in subprocess
    from algorithm_bot import AlgorithmBot

    try:
        # Create game
        game, game_config = create_game(task_id)
        game_name = game_config["game_name"]
        num_players = game.num_players()
        player_id = player_id % num_players

        # Get agent
        agent_class = GAME_AGENTS.get(game_name)
        if not agent_class:
            raise ValueError(f"No agent found for game: {game_name}")
        agent = agent_class()

        # Create AlgorithmBot
        bot = AlgorithmBot(
            game=game,
            player_id=player_id,
            agent=agent,
            algorithm=algorithm,
            seed=seed,
            mcts_simulations=mcts_simulations,
        )

        # Play game
        result = play_game(
            bot=bot,
            task_id=task_id,
            seed=seed,
            opponent=opponent,
            player_id=player_id,
        )
        return result

    except Exception as e:
        import traceback
        return {
            "task_id": task_id,
            "seed": seed,
            "score": 0.0,
            "success": False,
            "error": f"[{type(e).__name__}] {str(e)}\n{traceback.format_exc()}",
        }


def play_games_parallel(
    task_ids: list,
    seeds: list = None,
    algorithm: str = "mcts",
    opponent: str = "mcts",
    player_id: int = 0,
    max_workers: int = None,
    mcts_simulations: int = None,
    show_progress: bool = True,
) -> dict:
    """
    Play multiple games in parallel and compute average score.

    Args:
        task_ids: List of task IDs to play
        seeds: List of seeds (same length as task_ids). If None, random seeds.
        algorithm: Algorithm for your bot ("mcts", "random")
        opponent: Opponent type ("mcts", "random")
        player_id: Which position your bot plays
        max_workers: Number of parallel workers (default: CPU count)
        mcts_simulations: Override MCTS simulations (for faster testing)
        show_progress: Print progress updates

    Returns:
        dict with:
            - results: List of individual game results
            - avg_score: Average score across all games
            - win_rate: Percentage of games with score > 0.5
            - total_time: Total execution time
            - num_games: Number of games played
            - num_errors: Number of games with errors
    """
    from multiprocessing import Pool, cpu_count

    n_games = len(task_ids)
    if seeds is None:
        seeds = [random.randint(0, 2**32 - 1) for _ in range(n_games)]

    if len(seeds) != n_games:
        raise ValueError(f"seeds length ({len(seeds)}) must match task_ids length ({n_games})")

    if max_workers is None:
        max_workers = cpu_count()

    # Prepare arguments for workers
    worker_args = [
        {
            "task_id": task_id,
            "seed": seed,
            "algorithm": algorithm,
            "opponent": opponent,
            "player_id": player_id,
            "mcts_simulations": mcts_simulations,
        }
        for task_id, seed in zip(task_ids, seeds)
    ]

    start_time = time.time()
    results = []

    if show_progress:
        print(f"Running {n_games} games with {max_workers} workers...")

    # Run in parallel
    with Pool(processes=max_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_play_game_worker, worker_args)):
            results.append(result)
            if show_progress and (i + 1) % max(1, n_games // 10) == 0:
                print(f"  Progress: {i + 1}/{n_games} games completed")

    total_time = time.time() - start_time

    # Compute statistics
    scores = [r.get("score", 0.0) for r in results]
    errors = [r for r in results if "error" in r]

    avg_score = sum(scores) / len(scores) if scores else 0.0
    win_rate = sum(1 for s in scores if s > 0.5) / len(scores) if scores else 0.0

    if show_progress:
        print(f"\nCompleted {n_games} games in {total_time:.1f}s")
        print(f"Average score: {avg_score:.3f}")
        print(f"Win rate: {win_rate:.1%}")
        if errors:
            print(f"Errors: {len(errors)}")

    return {
        "results": results,
        "avg_score": avg_score,
        "win_rate": win_rate,
        "total_time": total_time,
        "num_games": n_games,
        "num_errors": len(errors),
    }

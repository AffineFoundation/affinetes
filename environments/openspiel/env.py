"""OpenSpiel Environment Actor"""

import os
import time
import random
import numpy as np
import asyncio
import concurrent.futures
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import mcts
import openai
import httpx
import pyspiel

from llm_bot import LLMBot
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
                llm_chat_fn=lambda messages: self._llm_chat(
                    messages, model, base_url, temperature, current_api_key, seed
                ),
                rng_seed=seed + 1,
                agent=agent,
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
                    bots.append(opponent_bot)

            loop = asyncio.get_event_loop()
            returns = await loop.run_in_executor(
                self.executor,
                evaluate_bots.evaluate_bots,
                game.new_initial_state(),
                bots,
                np.random.RandomState(seed),
            )

            llm_return = returns[llm_player_id]
            score = self._compute_score(returns, llm_player_id, game)

            return self._build_result(
                game_name=game_name,
                score=score,
                llm_return=llm_return,
                llm_player_id=llm_player_id,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
                conversation=llm_bot.get_conversation(),
                error=llm_bot.get_last_error(),
                usage=llm_bot.get_total_usage(),
                all_returns=returns,
            )

        except asyncio.TimeoutError:
            # Task timeout - return accumulated data
            return self._build_error_result(
                game_name=game_name,
                error=f"Task timeout exceeded ({task_timeout}s)",
                llm_bot=llm_bot,
                llm_player_id=llm_player_id,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
            )

        except Exception as e:
            # Other exceptions - return accumulated data with error details
            import traceback
            from llm_bot import ParsingError

            error_type = type(e).__name__
            
            # Special handling for ParsingError: treat as successful sampling with 0 score
            # The error is already recorded in conversation history by llm_bot
            if isinstance(e, ParsingError):
                print(f"[ParsingError] Game ended due to parsing failure - treating as valid sample with 0 score")
                return self._build_result(
                    game_name=game_name,
                    score=0.0,
                    llm_return=None,
                    llm_player_id=llm_player_id,
                    task_id=task_id,
                    seed=seed,
                    opponent=opponent,
                    start_time=start_time,
                    conversation=llm_bot.get_conversation() if llm_bot else [],
                    error=None,  # No error field - this is a valid sample
                    usage=llm_bot.get_total_usage() if llm_bot else None,
                    all_returns=None,
                )
            
            # Other exceptions: true errors
            # Try to get detailed error from llm_bot first
            if llm_bot and llm_bot.get_last_error():
                error_msg = llm_bot.get_last_error()
            else:
                error_msg = f"[{error_type}] {str(e)}\n{traceback.format_exc()}"

            return self._build_error_result(
                game_name=game_name,
                error=error_msg,
                llm_bot=llm_bot,
                llm_player_id=llm_player_id,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
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
            return mcts.MCTSBot(
                game=game,
                uct_c=1.414,
                max_simulations=max_simulations,
                evaluator=evaluator,
                random_state=np.random.RandomState(seed + 4),
            )
        else:
            raise ValueError(f"Unknown opponent type: {opponent}")

    def _build_error_result(
        self,
        game_name,
        error,
        llm_bot,
        llm_player_id,
        task_id,
        seed,
        opponent,
        start_time,
    ):
        """Build error result with accumulated data from llm_bot"""
        conversation = []
        usage = None
        
        if llm_bot is not None:
            try:
                conversation = llm_bot.get_conversation()
                usage = llm_bot.get_total_usage()
            except:
                pass
        
        return self._build_result(
            game_name=game_name,
            score=0.0,
            llm_return=-1.0,
            llm_player_id=llm_player_id,
            task_id=task_id,
            seed=seed,
            opponent=opponent,
            start_time=start_time,
            conversation=conversation,
            error=error,
            usage=usage,
            all_returns=None,
        )

    def _build_result(
        self,
        game_name,
        score,
        llm_return,
        llm_player_id,
        task_id,
        seed,
        opponent,
        start_time,
        conversation,
        error=None,
        usage=None,
        all_returns=None,
    ):
        """Build result dictionary"""
        result = {
            "task_name": f"openspiel:{game_name}",
            "score": score,
            "success": score > 0.5,
            "time_taken": time.time() - start_time,
            "extra": {
                "conversation": conversation,
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

        if error:
            # Error must be a string
            result["extra"]["error"] = str(error)

        return result

    async def _llm_chat(
        self, messages, model, base_url, temperature, current_api_key, seed=None
    ):
        """Call LLM API with streaming and message history support"""
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        async with openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=current_api_key,
            max_retries=0,
        ) as client:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            if seed is not None:
                params["seed"] = seed

            content_parts = []
            usage = None

            stream = await client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)

                if chunk.usage:
                    usage = chunk.usage.model_dump()

            if not content_parts:
                raise ValueError("LLM API returned empty content stream")

            content = "".join(content_parts)
            if not content:
                raise ValueError("LLM API returned None content")

            return content.strip(), usage

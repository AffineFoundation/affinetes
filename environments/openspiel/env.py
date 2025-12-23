"""OpenSpiel Environment Actor"""

import os
import time
import random
import numpy as np
import asyncio
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import mcts
import openai
import httpx
import pyspiel

from llm_bot import LLMBot
from game_config import create_game
from agents import GAME_AGENTS


class Actor:
    """OpenSpiel evaluation wrapper"""

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
        timeout: int = 600,
        temperature: float = 0.7,
        api_key: str = None,
        opponent: str = "random",
        task_timeout: int = 1800,
    ):
        """
        Run single game evaluation

        Args:
            task_id: Task identifier (12-digit format: GGGGCCCCCCCC)
            seed: Random seed for reproducibility
            model: LLM model name
            base_url: LLM API base URL
            timeout: API timeout in seconds (per LLM call)
            temperature: LLM temperature
            api_key: Override API key
            opponent: Opponent type ("random" or "mcts")
            task_timeout: Overall task timeout in seconds (default 1800s = 30min)
        """
        if task_id is None:
            task_id = random.randint(0, 10**11 - 1)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        current_api_key = api_key or self.api_key
        start_time = time.time()

        try:
            return await asyncio.wait_for(
                self._run_evaluation(
                    task_id,
                    seed,
                    model,
                    base_url,
                    timeout,
                    temperature,
                    current_api_key,
                    opponent,
                    start_time,
                ),
                timeout=task_timeout,
            )
        except asyncio.TimeoutError:
            return self._build_result(
                game_name="unknown",
                score=0.0,
                llm_return=-1.0,
                llm_player_id=seed % 2,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
                conversation=[],
                error=f"Task timeout exceeded ({task_timeout}s)",
            )

    async def _run_evaluation(
        self,
        task_id,
        seed,
        model,
        base_url,
        timeout,
        temperature,
        current_api_key,
        opponent,
        start_time,
    ):
        """Internal method to run evaluation (wrapped by timeout)"""
        llm_player_id = seed % 2

        try:
            game, game_config = create_game(task_id)

            num_players = game.num_players()
            
            # LLM always plays as player with id = llm_player_id % num_players
            llm_player_id = llm_player_id % num_players

            # Get agent for this game
            game_name = game_config["game_name"]
            agent_class = GAME_AGENTS.get(game_name)
            if not agent_class:
                raise ValueError(f"No agent found for game: {game_name}")
            
            agent = agent_class()

            llm_bot = LLMBot(
                game=game,
                player_id=llm_player_id,
                llm_chat_fn=lambda messages: self._llm_chat(
                    messages, model, base_url, timeout, temperature, current_api_key, seed
                ),
                rng_seed=seed + 1,
                agent=agent,
                max_parsing_retries=3,
            )

            # Create bots for all players
            bots = []
            for player_id in range(num_players):
                if player_id == llm_player_id:
                    bots.append(llm_bot)
                else:
                    # Other players use opponent bots with different seeds
                    opponent_bot = self._create_opponent_bot(
                        opponent, player_id, seed + 2 + player_id
                    )
                    bots.append(opponent_bot)

            returns = evaluate_bots.evaluate_bots(
                state=game.new_initial_state(),
                bots=bots,
                rng=np.random.RandomState(seed),
            )

            llm_return = returns[llm_player_id]
            # Compute score using game-type-aware logic
            score = self._compute_score(returns, llm_player_id, game)

            return self._build_result(
                game_name=game_config["game_name"],
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

        except Exception as e:
            import traceback

            return self._build_result(
                game_name="unknown",
                score=0.0,
                llm_return=-1.0,
                llm_player_id=llm_player_id,
                task_id=task_id,
                seed=seed,
                opponent=opponent,
                start_time=start_time,
                conversation=[],
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
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
                score = 0.5  # Degenerate case
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

    def _create_opponent_bot(self, opponent, player_id, seed):
        """Create opponent bot based on type"""
        if opponent == "random":
            return uniform_random.UniformRandomBot(
                player_id=player_id, rng=np.random.RandomState(seed + 2)
            )
        elif opponent == "mcts":
            evaluator = mcts.RandomRolloutEvaluator(
                n_rollouts=10, random_state=np.random.RandomState(seed + 3)
            )
            return mcts.MCTSBot(
                game=None,
                uct_c=2.0,
                max_simulations=100,
                evaluator=evaluator,
                random_state=np.random.RandomState(seed + 4),
            )
        else:
            raise ValueError(f"Unknown opponent type: {opponent}")

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
            # Error can be string or dict from llm_bot
            if isinstance(error, dict):
                result["extra"]["error"] = [
                    {"error": error.get("error"), "prompt": error.get("prompt")}
                ]
            else:
                result["extra"]["error"] = [{"error": error}]

        return result

    async def _llm_chat(
        self, messages, model, base_url, timeout, temperature, current_api_key, seed=None
    ):
        """Call LLM API with streaming and message history support"""
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        async with openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=current_api_key,
            timeout=httpx.Timeout(timeout),
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

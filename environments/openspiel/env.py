"""OpenSpiel Environment Actor

Supports two modes:
1. evaluate() - One-shot LLM evaluation with internal game loop
2. reset/step/stop - OpenEnv training interface for external control
"""

import os
import time
import random
import uuid
import re
import numpy as np
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import mcts
import pyspiel

from llm_bot import LLMBot
from game_config import create_game, decode_task_id
from agents import GAME_AGENTS
from base_agent import BaseGameAgent

# Import shared logging utilities
from request_logger import RequestLogger, log_event

from affinetes.core.openenv import OpenEnvResponse


@dataclass
class EpisodeState:
    """Training episode state for OpenSpiel games"""
    episode_id: str
    task_id: int
    seed: int
    game: Any  # pyspiel.Game
    state: Any  # pyspiel.State
    agent: BaseGameAgent
    llm_player_id: int
    opponent_type: str
    opponent_bot: Any  # pyspiel.Bot for opponent
    done: bool = False
    truncated: bool = False
    step_count: int = 0
    cumulative_reward: float = 0.0
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    last_opponent_action: Optional[int] = None
    rng: np.random.RandomState = field(default_factory=np.random.RandomState)


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
    """OpenSpiel evaluation wrapper with training support

    Provides two modes:
    1. evaluate() - One-shot LLM evaluation with internal game loop
    2. reset/step/stop - OpenEnv training interface for external control
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)

    def __init__(self, api_key: str = None):
        """
        Initialize Actor with API key

        Args:
            api_key: API key for LLM service. If not provided, uses CHUTES_API_KEY env var
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        # Training episode states - supports concurrent episodes
        self._episodes: Dict[str, EpisodeState] = {}
        self._last_observations: Dict[str, str] = {}

    # ========== Helper methods for training interface ==========

    def _to_python(self, val: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    def _info(self, ep: Optional[EpisodeState] = None, *, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build info dictionary for OpenEnv response"""
        info: Dict[str, Any] = {
            "task_id": self._to_python(ep.task_id) if ep else None,
            "seed": self._to_python(ep.seed) if ep else None,
            "game_name": ep.agent.game_name if ep else None,
            "llm_player_id": self._to_python(ep.llm_player_id) if ep else None,
            "opponent_type": ep.opponent_type if ep else None,
            "step_count": self._to_python(ep.step_count) if ep else 0,
            "cumulative_reward": self._to_python(ep.cumulative_reward) if ep else 0.0,
        }
        if ep and ep.last_opponent_action is not None:
            info["last_opponent_action"] = self._to_python(ep.last_opponent_action)
        if error:
            info["error"] = error
        return info

    def _resp(
        self,
        observation: str,
        *,
        episode_id: Optional[str] = None,
        reward: float = 0.0,
        done: bool = False,
        truncated: bool = False,
        info: Dict[str, Any],
    ) -> OpenEnvResponse:
        """Build OpenEnv response"""
        if episode_id:
            self._last_observations[episode_id] = observation
        return OpenEnvResponse(
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            episode_id=episode_id,
            info=info,
        )

    def _format_observation(self, ep: EpisodeState, include_legal_actions: bool = True) -> str:
        """Format current game state as observation text for LLM"""
        if not ep or ep.state.is_terminal():
            return self._format_terminal_observation(ep)

        # Get state description from agent
        state_desc = ep.agent.format_state(ep.state, ep.llm_player_id)

        # Build observation parts
        parts = [
            f"Game: {ep.agent.game_name}",
            f"You are Player {ep.llm_player_id}.",
            "",
            f"Current State:\n{state_desc}",
        ]

        # Add legal actions if requested and it's agent's turn
        if include_legal_actions:
            current_player = ep.state.current_player()
            if current_player == ep.llm_player_id:
                legal_actions = ep.state.legal_actions(ep.llm_player_id)
                actions_desc = []
                for action in legal_actions:
                    try:
                        action_str = ep.state.action_to_string(ep.llm_player_id, action)
                        actions_desc.append(f"  {action} -> {action_str}")
                    except:
                        actions_desc.append(f"  {action}")
                parts.append("")
                parts.append("Legal Actions:")
                parts.extend(actions_desc)
                parts.append("")
                parts.append("Your choice (action ID only):")
            else:
                parts.append("")
                parts.append(f"Waiting for Player {current_player} to move...")

        return "\n".join(parts)

    def _format_terminal_observation(self, ep: Optional[EpisodeState]) -> str:
        """Format terminal state observation"""
        if not ep:
            return "No active game."

        returns = ep.state.returns()
        llm_return = returns[ep.llm_player_id]
        score = self._compute_score(returns, ep.llm_player_id, ep.game)

        parts = [
            f"Game Over: {ep.agent.game_name}",
            f"You were Player {ep.llm_player_id}.",
            "",
            f"Final Returns: {returns}",
            f"Your Return: {llm_return}",
            f"Normalized Score: {score:.3f}",
        ]

        if score > 0.5:
            parts.append("Result: WIN")
        elif score < 0.5:
            parts.append("Result: LOSS")
        else:
            parts.append("Result: DRAW")

        return "\n".join(parts)

    def _parse_action(self, action_str: str, legal_actions: List[int]) -> Optional[int]:
        """Parse action ID from LLM response string"""
        # Try to extract number from the response
        numbers = re.findall(r'\d+', action_str.strip())
        if numbers:
            action_id = int(numbers[0])
            if action_id in legal_actions:
                return action_id
        return None

    def _auto_play_opponents(self, ep: EpisodeState) -> float:
        """
        Auto-play opponent turns until it's LLM player's turn or game ends.
        Returns accumulated reward from opponent moves.

        For SIMULTANEOUS games (e.g., goofspiel), stops immediately after
        chance nodes to let LLM provide their action via step().
        """
        if not ep or ep.state.is_terminal():
            return 0.0

        accumulated_reward = 0.0

        while not ep.state.is_terminal():
            current_player = ep.state.current_player()

            # Check for chance nodes
            if current_player == pyspiel.PlayerId.CHANCE:
                outcomes_with_probs = ep.state.chance_outcomes()
                if not outcomes_with_probs:
                    # Edge case: no chance outcomes (shouldn't happen but be safe)
                    break
                action_list, prob_list = zip(*outcomes_with_probs)
                action = ep.rng.choice(action_list, p=prob_list)
                ep.state.apply_action(action)
                continue

            # For SIMULTANEOUS games, stop and let LLM provide action via step()
            # In simultaneous games, both players act at once - we need LLM's input
            if current_player == pyspiel.PlayerId.SIMULTANEOUS:
                break

            # If it's LLM player's turn, stop
            if current_player == ep.llm_player_id:
                break

            # Opponent's turn - use opponent bot
            legal_actions = ep.state.legal_actions(current_player)
            if not legal_actions:
                break

            # Get opponent action
            if ep.opponent_bot:
                opp_action = ep.opponent_bot.step(ep.state)
            else:
                # Fallback to random
                opp_action = ep.rng.choice(legal_actions)

            ep.last_opponent_action = opp_action
            ep.state.apply_action(opp_action)
            ep.step_count += 1

            # Record opponent action
            try:
                action_str = ep.state.action_to_string(current_player, opp_action)
            except:
                action_str = str(opp_action)
            ep.action_history.append({
                "player": int(current_player),
                "action_id": int(opp_action),
                "action_str": action_str,
                "is_opponent": True,
            })

            # Get reward after opponent move (if any)
            if ep.state.rewards():
                step_reward = float(ep.state.rewards()[ep.llm_player_id])
                accumulated_reward += step_reward

        return accumulated_reward

    def _create_training_opponent_bot(self, opponent: str, player_id: int, seed: int, game, agent):
        """Create opponent bot for training mode"""
        game_type = game.get_type()

        # For simultaneous move games, MCTS doesn't work - fallback to random
        if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            return uniform_random.UniformRandomBot(
                player_id=player_id, rng=np.random.RandomState(seed + 2)
            )

        if opponent == "random":
            return uniform_random.UniformRandomBot(
                player_id=player_id, rng=np.random.RandomState(seed + 2)
            )
        elif opponent == "mcts":
            mcts_config = agent.get_mcts_config()
            if mcts_config is None:
                return uniform_random.UniformRandomBot(
                    player_id=player_id, rng=np.random.RandomState(seed + 2)
                )

            max_simulations, n_rollouts = mcts_config
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
            return TimedMCTSBot(mcts_bot)
        else:
            return uniform_random.UniformRandomBot(
                player_id=player_id, rng=np.random.RandomState(seed + 2)
            )

    # ========== OpenEnv Training Interface ==========

    async def reset(
        self,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        opponent: str = "mcts",
    ) -> OpenEnvResponse:
        """
        Reset environment and start a new game episode.

        Args:
            task_id: Task identifier (encodes game type and config)
            seed: Random seed for reproducibility
            opponent: Opponent type ("random" or "mcts", default: "mcts")

        Returns:
            OpenEnvResponse with initial observation
        """
        resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        resolved_task_id = int(task_id) if task_id is not None else random.randint(0, 10**11 - 1)
        opponent_type = opponent

        # Create game from task_id
        game, game_config = create_game(resolved_task_id)
        game_name = game_config["game_name"]

        # Get agent for this game
        agent_class = GAME_AGENTS.get(game_name)
        if not agent_class:
            return self._resp(
                f"Error: No agent found for game: {game_name}",
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "invalid_game", "message": f"Unknown game: {game_name}", "retryable": False}),
            )
        agent = agent_class()

        # Determine LLM player ID
        num_players = game.num_players()
        llm_player_id = resolved_seed % num_players

        # Create opponent bot (for other player(s))
        opponent_bot = None
        if num_players > 1:
            # For 2-player game, opponent is 1 - llm_player_id
            opponent_player_id = 1 - llm_player_id if num_players == 2 else 0
            opponent_bot = self._create_training_opponent_bot(
                opponent_type, opponent_player_id, resolved_seed, game, agent
            )

        # Create initial game state
        state = game.new_initial_state()

        # Generate episode ID
        episode_id = uuid.uuid4().hex

        # Create episode state
        ep = EpisodeState(
            episode_id=episode_id,
            task_id=resolved_task_id,
            seed=resolved_seed,
            game=game,
            state=state,
            agent=agent,
            llm_player_id=llm_player_id,
            opponent_type=opponent_type,
            opponent_bot=opponent_bot,
            rng=np.random.RandomState(resolved_seed + 100),
        )

        # Store in concurrent episodes dict
        self._episodes[episode_id] = ep

        # Auto-play opponent turns until it's LLM's turn
        self._auto_play_opponents(ep)

        # Check if game ended during opponent play
        if state.is_terminal():
            ep.done = True
            obs = self._format_terminal_observation(ep)
            returns = state.returns()
            final_reward = self._compute_score(returns, llm_player_id, game)
            return self._resp(
                obs,
                episode_id=episode_id,
                reward=final_reward,
                done=True,
                info=self._info(ep),
            )

        # Build initial observation with game rules
        rules = agent.get_rules()
        state_obs = self._format_observation(ep, include_legal_actions=True)

        initial_obs = f"""# Game Rules
{rules}

# Current Game State
{state_obs}"""

        return self._resp(initial_obs, episode_id=episode_id, info=self._info(ep))

    async def step(
        self,
        action: str,
        episode_id: Optional[str] = None,
    ) -> OpenEnvResponse:
        """
        Execute an action in the current game.

        Args:
            action: Action string (should contain action ID)
            episode_id: Episode identifier

        Returns:
            OpenEnvResponse with observation, reward, done status
        """
        # Validate episode_id is provided
        if not episode_id:
            return self._resp(
                "No episode_id provided. Call reset() first to get an episode_id.",
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "no_episode_id", "message": "episode_id is required for step().", "retryable": True}),
            )

        # Look up episode from concurrent episodes dict
        ep = self._episodes.get(episode_id)
        if not ep:
            return self._resp(
                f"Episode not found: {episode_id}. Call reset() to start a new episode.",
                episode_id=episode_id,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "episode_not_found", "message": f"Episode {episode_id} not found.", "retryable": True}),
            )

        # Check if episode already done
        if ep.done:
            obs = self._format_terminal_observation(ep)
            return self._resp(
                obs,
                episode_id=ep.episode_id,
                done=True,
                truncated=False,
                info=self._info(ep, error={"type": "episode_done", "message": "Episode already finished. Call reset().", "retryable": True}),
            )

        # Check if it's LLM player's turn
        current_player = ep.state.current_player()

        # Handle SIMULTANEOUS games (e.g., goofspiel) - both players act at once
        is_simultaneous = (current_player == pyspiel.PlayerId.SIMULTANEOUS)

        if not is_simultaneous and current_player != ep.llm_player_id:
            # For sequential games, check if it's actually LLM's turn
            return self._resp(
                f"Not your turn. Current player: {current_player}",
                episode_id=ep.episode_id,
                info=self._info(ep, error={"type": "not_your_turn", "message": f"Current player is {current_player}", "retryable": False}),
            )

        # Get legal actions for LLM player
        legal_actions = ep.state.legal_actions(ep.llm_player_id)
        if not legal_actions:
            # No legal actions - game should end
            ep.done = True
            obs = self._format_terminal_observation(ep)
            returns = ep.state.returns()
            final_reward = self._compute_score(returns, ep.llm_player_id, ep.game)
            return self._resp(obs, episode_id=ep.episode_id, reward=final_reward, done=True, info=self._info(ep))

        # Parse action from string
        parsed_action = self._parse_action(action, legal_actions)
        if parsed_action is None:
            # Invalid action - return error but don't end episode
            actions_desc = []
            for a in legal_actions[:10]:  # Show first 10 actions
                try:
                    a_str = ep.state.action_to_string(ep.llm_player_id, a)
                    actions_desc.append(f"  {a} -> {a_str}")
                except:
                    actions_desc.append(f"  {a}")
            if len(legal_actions) > 10:
                actions_desc.append(f"  ... and {len(legal_actions) - 10} more")

            obs = f"""Invalid action: "{action}"

Could not parse a valid action ID. Please respond with just the action ID number.

Legal Actions:
{chr(10).join(actions_desc)}

Your choice (action ID only):"""
            return self._resp(
                obs,
                episode_id=ep.episode_id,
                reward=-0.01,  # Small penalty for invalid action
                info=self._info(ep, error={"type": "invalid_action", "message": "Could not parse action ID", "retryable": True}),
            )

        # Apply the action
        try:
            action_str = ep.state.action_to_string(ep.llm_player_id, parsed_action)
        except:
            action_str = str(parsed_action)

        if is_simultaneous:
            # SIMULTANEOUS games: combine LLM action with opponent action
            # Joint action encoding: joint = p0_action + p1_action * num_actions
            num_actions = ep.game.num_distinct_actions()

            # Get opponent action from bot
            opponent_action = 0
            opponent_legal = ep.state.legal_actions(1 - ep.llm_player_id)
            if opponent_legal:
                if ep.opponent_bot:
                    opponent_action = ep.opponent_bot.step(ep.state)
                else:
                    opponent_action = ep.rng.choice(opponent_legal)

            # Compute joint action based on player positions
            if ep.llm_player_id == 0:
                joint_action = parsed_action + opponent_action * num_actions
            else:
                joint_action = opponent_action + parsed_action * num_actions

            # Apply joint action
            ep.state.apply_action(joint_action)
            ep.last_opponent_action = opponent_action

            # Record opponent action
            try:
                opp_action_str = ep.state.action_to_string(1 - ep.llm_player_id, opponent_action)
            except:
                opp_action_str = str(opponent_action)
            ep.action_history.append({
                "player": int(1 - ep.llm_player_id),
                "action_id": int(opponent_action),
                "action_str": opp_action_str,
                "is_opponent": True,
            })
        else:
            # Sequential game: just apply LLM's action
            ep.state.apply_action(parsed_action)

        ep.step_count += 1

        # Record LLM action
        ep.action_history.append({
            "player": int(ep.llm_player_id),
            "action_id": int(parsed_action),
            "action_str": action_str,
            "is_opponent": False,
        })

        # Get immediate reward (if available)
        step_reward = 0.0
        if ep.state.rewards():
            step_reward = float(ep.state.rewards()[ep.llm_player_id])

        # Check if game ended after our move
        if ep.state.is_terminal():
            ep.done = True
            returns = ep.state.returns()
            final_score = self._compute_score(returns, ep.llm_player_id, ep.game)
            ep.cumulative_reward += final_score
            obs = self._format_terminal_observation(ep)
            return self._resp(obs, episode_id=ep.episode_id, reward=final_score, done=True, info=self._info(ep))

        # Auto-play opponent turns
        opponent_reward = self._auto_play_opponents(ep)
        step_reward += opponent_reward
        ep.cumulative_reward += step_reward

        # Check if game ended after opponent moves
        if ep.state.is_terminal():
            ep.done = True
            returns = ep.state.returns()
            final_score = self._compute_score(returns, ep.llm_player_id, ep.game)
            obs = self._format_terminal_observation(ep)
            return self._resp(obs, episode_id=ep.episode_id, reward=final_score, done=True, info=self._info(ep))

        # Game continues - return current state
        obs = self._format_observation(ep, include_legal_actions=True)
        return self._resp(obs, episode_id=ep.episode_id, reward=step_reward, info=self._info(ep))

    async def state(
        self,
        episode_id: Optional[str] = None,
    ) -> OpenEnvResponse:
        """
        Get current game state without advancing (no state transition).

        Args:
            episode_id: Episode identifier

        Returns:
            OpenEnvResponse with current observation
        """
        if not episode_id:
            return self._resp(
                "No episode_id provided. Call reset() first to get an episode_id.",
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "no_episode_id", "message": "episode_id is required.", "retryable": True}),
            )

        ep = self._episodes.get(episode_id)
        if not ep:
            obs = self._last_observations.get(episode_id, "Episode not found.")
            return self._resp(
                obs,
                episode_id=episode_id,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "episode_not_found", "message": f"Episode {episode_id} not found.", "retryable": True}),
            )

        obs = self._format_observation(ep, include_legal_actions=True) if not ep.done else self._format_terminal_observation(ep)
        return self._resp(obs, episode_id=ep.episode_id, done=ep.done, truncated=ep.truncated, info=self._info(ep))

    async def stop(
        self,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stop (terminate) the active episode and release resources.

        Args:
            episode_id: Episode identifier

        Returns:
            Status dictionary
        """
        if not episode_id:
            return {"status": "ok", "stopped": False, "message": "No episode_id provided"}

        # Remove from concurrent episodes dict
        ep = self._episodes.pop(episode_id, None)
        self._last_observations.pop(episode_id, None)

        if not ep:
            return {"status": "ok", "stopped": False, "message": f"Episode {episode_id} not found"}

        return {"status": "ok", "stopped": True, "episode_id": episode_id}

    # ========== Original Evaluation Interface ==========

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

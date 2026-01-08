"""Algorithm Bot for trajectory generation with LLM-format conversation recording"""

import pyspiel
import numpy as np
from typing import Optional, Dict, List, Any

from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import mcts

from base_agent import BaseGameAgent


class SafeRandomRolloutEvaluator(mcts.Evaluator):
    """Safe MCTS evaluator that handles edge cases"""
    
    def __init__(self, n_rollouts=1, random_state=None):
        self._n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()
    
    def evaluate(self, state):
        if state.is_terminal():
            return state.returns()
        
        legal_actions = state.legal_actions()
        if not legal_actions:
            return state.returns()
        
        total_returns = np.zeros(state.num_players())
        
        for _ in range(self._n_rollouts):
            working_state = state.clone()
            
            while not working_state.is_terminal():
                legal_actions = working_state.legal_actions()
                if not legal_actions:
                    break
                action = self._random_state.choice(legal_actions)
                working_state.apply_action(action)
            
            total_returns += working_state.returns()
        
        return total_returns / self._n_rollouts
    
    def prior(self, state):
        legal_actions = state.legal_actions()
        if not legal_actions:
            return []
        prob = 1.0 / len(legal_actions)
        return [(action, prob) for action in legal_actions]


class AlgorithmBot(pyspiel.Bot):
    """
    Wraps game algorithms (MCTS, random, etc.) and records moves in LLM conversation format.
    
    Used for generating training trajectories for LLM fine-tuning.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        agent: BaseGameAgent,
        algorithm: str = "mcts",
        seed: Optional[int] = None,
        mcts_simulations: Optional[int] = None,
        mcts_rollouts: int = 1,
    ):
        """
        Initialize Algorithm Bot
        
        Args:
            game: pyspiel.Game instance
            player_id: Player ID (0, 1, ...)
            agent: BaseGameAgent for game-specific formatting
            algorithm: Algorithm type ("mcts", "random")
            seed: Random seed for reproducibility
            mcts_simulations: Override MCTS simulations (default: from agent config)
            mcts_rollouts: Number of rollouts per MCTS evaluation
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._agent = agent
        self._algorithm = algorithm
        self._seed = seed or 42
        self._rng = np.random.RandomState(self._seed)
        
        # Create inner bot based on algorithm
        self._inner_bot = self._create_inner_bot(mcts_simulations, mcts_rollouts)
        
        # Conversation history (LLM format for training)
        self._conversation: List[Dict[str, str]] = []
        self._action_history: List[Dict[str, Any]] = []
        self._system_prompt_generated = False
        self._observation: Optional[str] = None

    def _create_inner_bot(self, mcts_simulations: Optional[int], mcts_rollouts: int):
        """Create the underlying algorithm bot"""
        if self._algorithm == "random":
            return uniform_random.UniformRandomBot(
                player_id=self._player_id,
                rng=np.random.RandomState(self._seed + 1)
            )
        elif self._algorithm == "mcts":
            # Get MCTS config from agent or use override
            mcts_config = self._agent.get_mcts_config()
            if mcts_config is None:
                # Fallback to random if MCTS not supported
                return uniform_random.UniformRandomBot(
                    player_id=self._player_id,
                    rng=np.random.RandomState(self._seed + 1)
                )
            
            max_sims, n_rollouts = mcts_config
            if mcts_simulations is not None:
                max_sims = mcts_simulations
            if mcts_rollouts != 1:
                n_rollouts = mcts_rollouts
            
            evaluator = SafeRandomRolloutEvaluator(
                n_rollouts=n_rollouts,
                random_state=np.random.RandomState(self._seed + 2)
            )
            return mcts.MCTSBot(
                game=self._game,
                uct_c=1.414,
                max_simulations=max_sims,
                evaluator=evaluator,
                random_state=np.random.RandomState(self._seed + 3),
            )
        else:
            raise ValueError(f"Unknown algorithm: {self._algorithm}")

    def restart_at(self, state):
        """Reset for new game"""
        self._conversation.clear()
        self._action_history.clear()
        self._system_prompt_generated = False
        self._observation = None
        if hasattr(self._inner_bot, 'restart_at'):
            self._inner_bot.restart_at(state)

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
            "is_algorithm": bool(player_id == self._player_id)
        })

        try:
            self._observation = state.observation_string()
        except:
            try:
                self._observation = str(state)
            except:
                self._observation = None
        
        if hasattr(self._inner_bot, 'inform_action'):
            self._inner_bot.inform_action(state, player_id, action)

    def step(self, state) -> int:
        """
        Get action from algorithm and record in conversation format.

        Generates prompts identical to LLMBot for training data consistency.
        """
        # Generate system prompt (first time only)
        if not self._system_prompt_generated:
            system_prompt = self._agent.generate_system_prompt()
            self._conversation.append({"role": "system", "content": system_prompt})
            self._system_prompt_generated = True

        # Get legal actions
        legal_actions = state.legal_actions(self._player_id)

        # Generate user prompt (same as LLMBot)
        user_prompt = self._agent.generate_user_prompt(
            state=state,
            player_id=self._player_id,
            legal_actions=legal_actions
        )
        self._conversation.append({"role": "user", "content": user_prompt})

        # Get action from inner algorithm
        action = self._inner_bot.step(state)

        # Record as assistant response (just the action ID, like LLM output)
        self._conversation.append({"role": "assistant", "content": str(action)})

        # Record in action history
        self.inform_action(state, self._player_id, action)

        return action

    def get_conversation(self) -> List[Dict[str, str]]:
        """Return conversation history in LLM format for training"""
        return self._conversation.copy()

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Return action history"""
        return self._action_history.copy()

    def get_observation(self) -> Optional[str]:
        """Return last observation"""
        return self._observation

    def get_algorithm(self) -> str:
        """Return algorithm name"""
        return self._algorithm


"""Game configuration and task_id decoding logic

Uses Agent system for parameter generation.
Each game's Agent class provides generate_params() method.
"""

import pyspiel
from typing import Dict, Any
from agents import GAME_AGENTS


# Game order - Prioritized by evaluation quality and model capability assessment
# 
# Quality Criteria:
# 1. Trajectory Diversity: task_id + seed generates many different game trajectories (防止模型背题)
# 2. Moderate Difficulty: No guaranteed winning strategy, requires reasoning
# 3. High Interactivity: Multi-step decision making, not single choices
# 4. Reproducibility: Same task_id + seed produces identical trajectory
#
# Removed Games (with reasons):
# - "breakthrough": 100% success rate, too easy, no evaluation discrimination
# - "pig": Limited strategic depth, mostly luck-based (roll or hold)
# - "phantom_ttt": Even with hidden info, 3x3 grid limits complexity
# - "cribbage": 0% success rate, rules too complex for LLM to understand
# - "battleship": 979k avg tokens (100% success but economically prohibitive)
#
AVAILABLE_GAMES = [
    # Tier 1: Excellent evaluation games (⭐⭐⭐⭐⭐)
    # High trajectory diversity, strong strategic depth, proven evaluation quality
    "goofspiel",        # idx=0:  Bidding strategy, 100% success, 7.8k tokens, high diversity
    "liars_dice",       # idx=1:  Probability reasoning, 100% success, 1.1k tokens ⭐
    "leduc_poker",      # idx=2:  Poker reasoning, 100% success, 1.3k tokens ⭐
    "gin_rummy",        # idx=3:  Card strategy, 100% success, 167.8k tokens (acceptable)
    
    # Tier 2: High-quality evaluation games (⭐⭐⭐⭐)
    # Good trajectory diversity, moderate difficulty, effective evaluation
    "othello",          # idx=4:  Spatial reasoning, 50% success, 105.8k tokens
    "backgammon",       # idx=5:  Long-term planning, 50% success, 347.2k tokens
    "hex",              # idx=6:  Path planning, 100% success, 13.9k tokens
    "clobber",          # idx=7:  Capture tactics, 100% success, 16.9k tokens (FIXED)
    
    # Tier 3: Multi-player & Complex games (⭐⭐⭐⭐)
    # Higher complexity, useful for advanced evaluation
    "hearts",           # idx=8:  4-player card game, 50% success, 27.5k tokens (FIXED)
    "euchre",           # idx=9:  Trump-based card game, 100% success, 5.8k tokens
    "dots_and_boxes",   # idx=10: Spatial control, 100% success, 62.1k tokens
    
    # Tier 4: High-complexity strategy games (⭐⭐⭐⭐)
    # Complex but high token consumption - use for advanced testing
    "go",               # idx=11: Board strategy (5x5/7x7), 50% success, 119.1k tokens (OPTIMIZED)
    "chess",            # idx=12: Complex strategy, 100% success, 287.1k tokens (OPTIMIZED)
    "checkers",         # idx=13: Classic strategy, 100% success, 83.8k tokens
    "quoridor",         # idx=14: Path blocking, 100% success, 38.9k tokens
    
    # Tier 5: Probability & Imperfect Information (⭐⭐⭐)
    # Special category for testing hidden information reasoning
    "blackjack",        # idx=15: Probability reasoning (vs dealer), 50% success, 519 tokens (FIXED)
    "phantom_ttt",      # idx=16: Hidden opponent moves, 100% success, 1.2k tokens (kept for imperfect info testing)
    
    # Tier 6: Single-player games (⭐⭐⭐⭐⭐) - NEW
    # High trajectory diversity, excellent for testing spatial/sequential reasoning
    "2048",             # idx=17: Sliding tile puzzle, spatial planning, ~50-200 steps
    "solitaire",        # idx=18: Card sequencing, shuffled deck (high diversity), ~30-100 steps
    
    # Tier 7: Advanced strategy games (⭐⭐⭐⭐⭐) - NEW ADDITIONS
    # High difficulty, complex strategy space, excellent quality
    "bridge",           # idx=19: 4-player card game, bidding & trick-taking, imperfect info
    "amazons",          # idx=20: Territorial control, queen-like moves + arrow placement
    "oware",            # idx=21: Mancala variant, seed counting & capturing strategy
]

# Notes on game changes:
# 1. Breakthrough: 100% success → no discrimination between models
# 2. Pig: Too simple, luck-dominant (just "roll or hold")
# 3. Cribbage: 0% success even after rule improvements
# 4. Battleship: 979k tokens unsustainable (despite 100% success)



def decode_task_id(task_id: int) -> Dict[str, Any]:
    """
    Decode task_id into game configuration
    
    task_id format: GGGGCCCCCCCC (12-digit integer)
    - GGGG: Game index (4 digits, 0-9999)
    - CCCCCCCC: Configuration variant (8 digits, 0-99999999)
    
    Args:
        task_id: 12-digit integer representing game and configuration
        
    Returns:
        Dictionary with:
        - game_name: str
        - game_idx: int
        - config_id: int
        - game_params: dict
        
    Examples:
        task_id = 0 -> kuhn_poker with default config
        task_id = 100000000 -> leduc_poker with default config
        task_id = 200000002 -> liars_dice with 3 dice per player
        
    Note:
        AVAILABLE_GAMES order is stable - new games are always appended.
        This ensures existing task_ids always map to the same game.
    """
    game_idx = task_id // 100000000
    config_id = task_id % 100000000
    game_name = AVAILABLE_GAMES[game_idx % len(AVAILABLE_GAMES)]
    game_params = generate_game_params(game_name, config_id)
    
    return {
        "game_name": game_name,
        "game_idx": game_idx,
        "config_id": config_id,
        "game_params": game_params
    }


def generate_game_params(game_name: str, config_id: int) -> Dict[str, Any]:
    """
    Generate game parameter variants based on config_id
    
    Uses Agent's generate_params() method for each game.
    
    Args:
        game_name: Name of the game
        config_id: 8-digit configuration variant ID
        
    Returns:
        Dictionary of game parameters
    """
    agent_class = GAME_AGENTS.get(game_name)
    if not agent_class:
        raise ValueError(f"No agent found for game: {game_name}")
    
    agent = agent_class()
    return agent.generate_params(config_id)


def create_game(task_id: int):
    """
    Create game instance from task_id
    
    Args:
        task_id: Task identifier
        
    Returns:
        Tuple of (game, config_dict)
    """
    config = decode_task_id(task_id)
    
    game = pyspiel.load_game(
        config["game_name"],
        config["game_params"]
    )
    
    return game, config


def get_game_info():
    """
    Get information about all available games
    
    Returns:
        List of dictionaries with game info
    """
    info = []
    for idx, game_name in enumerate(AVAILABLE_GAMES):
        info.append({
            "idx": idx,
            "name": game_name,
            "task_id_start": idx * 100000000,
        })
    
    return info
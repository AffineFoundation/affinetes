"""Game configuration and task_id decoding logic

Uses Agent system for parameter generation.
Each game's Agent class provides generate_params() method.
"""

import pyspiel
from typing import Dict, Any
from agents import GAME_AGENTS


# Game order - Prioritized by evaluation quality and model capability assessment
# High-quality games (effective evaluation, good interaction, reasonable challenge) first
# Lower-quality games (too easy, poor interaction, or limited scope) last
AVAILABLE_GAMES = [
    # Tier 1: Excellent evaluation games (⭐⭐⭐⭐⭐)
    "goofspiel",        # idx=0:  Best token efficiency, tests bidding strategy, 100% success but meaningful
    "liars_dice",       # idx=1:  Excellent interaction design, tests probability reasoning
    "leduc_poker",      # idx=2:  High efficiency, tests poker reasoning, 80% success (good distinction)
    "gin_rummy",        # idx=3:  Great interaction quality, tests card game strategy
    
    # Tier 2: Good evaluation games (⭐⭐⭐⭐)
    "othello",          # idx=4:  Clear visualization, 40% success, tests spatial reasoning
    "backgammon",       # idx=5:  Tests long-term planning, 40% success, high token but acceptable
    "hex",              # idx=6:  Tests path planning, 60% success (improved with format_state)
    
    # Tier 3: Acceptable but needs improvement (⭐⭐⭐)
    "battleship",       # idx=7:  Good concept (imperfect info, dual-phase), FIXED token issue
    "blackjack",        # idx=8:  Good interaction, FIXED seed diversity issue
    
    # Tier 4: Limited evaluation value (⭐⭐)
    "breakthrough",     # idx=9:  100% success (too easy), good interaction but weak opponent
    "pig",              # idx=10: Simple dice game, limited strategic depth
    "phantom_ttt",      # idx=11: Tic-tac-toe variant, limited complexity
    
    # Tier 5: Complex games needing further testing
    "hearts",           # idx=12: Multi-player card game, untested
    "cribbage",         # idx=13: Complex scoring, untested
    "euchre",           # idx=14: Trump-based card game, untested
    "go",               # idx=15: Extremely complex, may have scalability issues
    "chess",            # idx=16: Very complex, long games, high token consumption expected
    "checkers",         # idx=17: Classic strategy game, untested
    "dots_and_boxes",   # idx=18: Simple but requires planning, untested
    "clobber",          # idx=19: Untested
    "quoridor",         # idx=20: Untested
]



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
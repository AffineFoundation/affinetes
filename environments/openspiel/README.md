# OpenSpiel Environment

OpenSpiel game-playing evaluation environment for LLM agents.

## Overview

This environment wraps [OpenSpiel](https://github.com/google-deepmind/open_spiel) games to evaluate LLM's strategic game-playing abilities. The LLM plays against built-in opponents (random or MCTS) in various games.

## Design Philosophy

**Maximize reuse of OpenSpiel's native tools** - This wrapper only handles:
- task_id decoding to game configuration
- LLM as a `pyspiel.Bot` implementation
- Result format conversion

Everything else (game loop, state description, opponent bots) is reused from OpenSpiel.

## Architecture

```
affinetes/environments/openspiel/
├── env.py              # Actor class - evaluation entry point
├── llm_bot.py          # LLMBot - implements pyspiel.Bot interface
├── game_config.py      # task_id decoding logic
├── models.py           # Data models
├── requirements.txt    
├── Dockerfile
└── README.md
```

## Usage

```python
from env import Actor

actor = Actor(api_key="...")

# Basic evaluation
result = await actor.evaluate(
    task_id=12345,
    seed=42,
    opponent="random"
)
print(f"Score: {result['score']}")  # 0.0-1.0

# Use MCTS opponent (harder)
result = await actor.evaluate(
    task_id=12345,
    seed=42,
    opponent="mcts"
)

# Multi-player game (LLM plays one position, bots fill others)
result = await actor.evaluate(
    task_id=1100000000,  # Hearts (4-player)
    seed=42,
    opponent="random"
)
```

## task_id Format

```
task_id = GGGGCCCCCCCC (12-digit integer)
GGGG: Game index (4 digits, 0-9999)
CCCCCCCC: Configuration variant (8 digits, 0-99999999)
```

### Examples

```python
# Leduc Poker, default config
task_id = 0

# Liar's Dice, 3 dice per player
task_id = 100000002

# Battleship, 10x10 board with 4 ships
task_id = 200000004

# Hearts, variant with pass_cards enabled
task_id = 1100000001
```

## Supported Games (20 Games Total)

**SELECTION CRITERIA (2025-12-20)**:
- ✅ **MUST have ≥100 distinct trajectories per task_id+seed combination**
- ✅ **Models CANNOT memorize a single winning strategy**
- ✅ **Balanced mix**: 11 high-randomness games + 9 large deterministic games

### Category A: High-Randomness Games (11 games)

These games use explicit randomness (dice, cards, ship placement) to prevent memorization.

#### Game 0: Leduc Poker
- **Type**: 2-player, imperfect information
- **Randomness**: Card dealing from 6-card deck
- **Trajectories**: ~120 per config
- **Config variants**: 1
- **Why kept**: Meets ≥100 threshold

#### Game 1: Liar's Dice
- **Type**: 2-player, imperfect information
- **Randomness**: Dice rolls (1-5 dice per player)
- **Trajectories**: 36 to 60M per config
- **Config variants**: 5
- **Total**: ~62 million trajectories

#### Game 2: Battleship
- **Type**: 2-player, imperfect information
- **Randomness**: Ship placement
- **Trajectories**: 10¹² to 10²⁰ per config
- **Config variants**: 9 (3 board sizes × 3 ship counts)
- **Total**: > 10²⁰ trajectories

#### Game 3: Goofspiel
- **Type**: 2-player, perfect info with random prizes
- **Randomness**: Prize card permutation
- **Trajectories**: 40K to 20T per config
- **Config variants**: 5 (8-16 cards)
- **Total**: > 20 trillion trajectories

#### Game 4: Gin Rummy
- **Type**: 2-player, card game
- **Randomness**: 52-card deck dealing
- **Trajectories**: ~10⁶⁰ per config
- **Config variants**: 9 (3 hand sizes × 3 knock values)
- **Total**: > 10⁶⁰ trajectories

#### Game 5: Backgammon
- **Type**: 2-player, classic dice game
- **Randomness**: Two 6-sided dice per turn
- **Trajectories**: > 10⁵⁰
- **Config variants**: 1

#### Game 6: Pig
- **Type**: 2-player, dice game
- **Randomness**: Single die per roll
- **Trajectories**: > 10,000 per config
- **Config variants**: 3 (different win scores)

#### Game 7: Blackjack
- **Type**: 1-player vs dealer
- **Randomness**: 52-card deck
- **Trajectories**: > 10⁶⁰
- **Config variants**: 1

#### Game 8: Phantom Tic-Tac-Toe
- **Type**: 2-player, imperfect information
- **Randomness**: Hidden opponent moves
- **Trajectories**: > 1,000 per config
- **Config variants**: 2

#### Game 9: Breakthrough
- **Type**: 2-player, board game
- **Randomness**: Seed-based random initial state
- **Trajectories**: > 100 per config
- **Config variants**: 2 (6x6, 8x8 boards)

#### Game 10: Hex
- **Type**: 2-player, connection game
- **Randomness**: Different board sizes create diverse openings
- **Trajectories**: > 100 per config
- **Config variants**: 4 (5x5, 7x7, 9x9, 11x11)

---

### Category B: Large Deterministic Games (9 games)

These games are deterministic but have MASSIVE strategy spaces that prevent single-strategy memorization.

**Why kept**: Different parameter variants (e.g., Go 7x7 vs 11x11, Chess vs Checkers) require fundamentally different strategies. Models cannot memorize universal winning strategies.

**Why 3 small games were REMOVED**: Tic-Tac-Toe 3x3, Connect Four, and Nim can be completely solved with memorizable optimal strategies.

#### Game 11: Hearts (4-player card game)
- **Type**: 4-player, card game
- **Randomness**: 52-card deck dealing
- **Trajectories**: > 10⁶⁰
- **Config variants**: 8 (rule combinations)
- **Strategy space**: Cannot memorize optimal play due to card randomness

#### Game 12: Cribbage (2-4 player card game)
- **Type**: 2-4 players, card game
- **Randomness**: 52-card deck dealing
- **Trajectories**: > 10⁶⁰
- **Config variants**: 3 (2, 3, 4 players)
- **Strategy space**: Different player counts require different strategies

#### Game 13: Euchre (4-player card game)
- **Type**: 4-player, card game
- **Randomness**: 24-card deck dealing
- **Trajectories**: > 10²⁰
- **Config variants**: 4 (rule combinations)
- **Strategy space**: Trump suit randomness prevents memorization

#### Game 14: Othello (Reversi)
- **Type**: 2-player, board game
- **Randomness**: None (deterministic)
- **Trajectories**: > 100 per config
- **Config variants**: 2 (6x6, 8x8 boards)
- **Strategy space**: > 10⁴⁰ (different board sizes need different strategies)

#### Game 15: Go
- **Type**: 2-player, ancient board game
- **Randomness**: None (deterministic)
- **Trajectories**: > 100 per config
- **Config variants**: 9 (3 sizes × 3 komi values)
- **Strategy space**: > 10¹⁰⁰ (different sizes require completely different opening theory)

#### Game 16: Chess
- **Type**: 2-player, classic strategy game
- **Randomness**: None (deterministic)
- **Trajectories**: Effectively infinite due to complexity
- **Config variants**: 1
- **Strategy space**: > 10⁴⁰ (cannot memorize all positions)

#### Game 17: Checkers
- **Type**: 2-player, board game
- **Randomness**: None (deterministic)
- **Trajectories**: > 10²⁰
- **Config variants**: 1
- **Strategy space**: > 10²⁰ (weakly solved but still complex)

#### Game 18: Dots and Boxes
- **Type**: 2-player, pencil-and-paper game
- **Randomness**: None (deterministic)
- **Trajectories**: > 100 per config
- **Config variants**: 3 (3x3, 4x4, 5x5 grids)
- **Strategy space**: Different grid sizes require different endgame play

#### Game 19: Clobber
- **Type**: 2-player, board game
- **Randomness**: None (deterministic)
- **Trajectories**: > 100 per config
- **Config variants**: 3 (5x5, 6x6, 7x7 boards)
- **Strategy space**: Different sizes change opening dynamics

#### Game 20: Quoridor
- **Type**: 2-player, path-blocking game
- **Randomness**: None (deterministic)
- **Trajectories**: > 100 per config
- **Config variants**: 4 (2 sizes × 2 wall counts)
- **Strategy space**: Different wall counts create different strategic priorities

---

## Removed Games (3 small deterministic games)

**Reason**: These games can be solved with a single memorizable strategy, violating the "no universal winning strategy" requirement.

- ❌ **Tic-Tac-Toe 3x3**: Completely solved, guaranteed draw with optimal play
- ❌ **Connect Four**: Weakly solved, first player has known winning strategy
- ❌ **Nim**: Mathematically solvable with nimbers formula

**Replaced with**: Hearts, Cribbage, Euchre (high-randomness card games)

## Multi-Player Game Support

The environment now supports games with 2-4 players:
- **LLM plays one position** (determined by `seed % num_players`)
- **Bots fill remaining positions** using the specified opponent type
- **Examples**: Hearts (4p), Cribbage (2-4p), Euchre (4p)

```python
# Example: LLM plays Hearts as player 0, 3 random bots as players 1, 2, 3
result = await actor.evaluate(
    task_id=1100000000,  # Hearts
    seed=42,  # seed % 4 = 2, so LLM plays as player 2
    opponent="random"
)
```

## Reproducibility & Trajectory Coverage

All randomness is controlled by `seed`:
- `llm_player_id = seed % num_players`
- Game RNG seed (`rng=np.random.RandomState(seed)`)
- Opponent RNG seeds (derived: seed+2, seed+3, seed+4, ...)
- LLM API seed (same seed)

**Same `task_id + seed` combination always produces the same game.**

### Trajectory Space Summary

| Category | Games | Total Trajectories | Memorizable? |
|----------|-------|-------------------|--------------|
| High-Randomness | 11 | > 10⁶⁰ | ❌ No |
| Large Deterministic | 9 | > 10⁴⁰ strategy space | ❌ No |
| **TOTAL** | **20** | **> 10⁶⁰** | **❌ No** |

All games meet the **≥100 trajectories** requirement and prevent strategy memorization.

## Key Components

### LLMBot (`llm_bot.py`)

Wraps LLM as an OpenSpiel `pyspiel.Bot`:
- `step(state)`: Generate LLM prompt, call API, parse action
- `restart_at(state)`: Reset for new game
- `inform_action(...)`: Track opponent moves

### Game Config (`game_config.py`)

Decodes `task_id` to game configuration:
- Game selection (circular indexing)
- Parameter variants (board size, player count, etc.)

### Actor (`env.py`)

Main evaluation entry point:
1. Create game from task_id
2. Determine LLM player ID from seed
3. Create LLMBot + OpponentBots for all positions
4. Call OpenSpiel's `evaluate_bots()` for gameplay
5. Return normalized score (0-1)

## Dependencies

- `open-spiel>=1.0.0` - Game engine
- `numpy>=1.24.0` - Random number generation
- `openai>=1.0.0` - LLM API client
- `httpx>=0.24.0` - HTTP client
- `pydantic>=2.0.0` - Data models

## Design Advantages

1. **Code Simplicity**: Minimal custom code
2. **Ecosystem Reuse**: Leverages OpenSpiel's battle-tested infrastructure
3. **Extensibility**: Add new games by appending to `AVAILABLE_GAMES`
4. **Anti-Memorization**: 20 diverse games prevent strategy memorization
5. **Multi-Player Support**: Handles 2-4 player games seamlessly

## References

- [OpenSpiel Documentation](https://openspiel.readthedocs.io/)
- [OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel)
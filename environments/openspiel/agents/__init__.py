"""OpenSpiel Game Agents Registry"""

# Import all game agents (21 original + 2 single-player)
from .liars_dice_agent import LiarsDiceAgent
from .leduc_poker_agent import LeducPokerAgent
from .battleship_agent import BattleshipAgent
from .backgammon import BackgammonAgent
from .pig import PigAgent
from .goofspiel import GoofspielAgent
from .gin_rummy import GinRummyAgent
from .blackjack import BlackjackAgent
from .phantom_ttt import PhantomTttAgent
from .breakthrough import BreakthroughAgent
from .hex import HexAgent
from .hearts import HeartsAgent
from .cribbage import CribbageAgent
from .euchre import EuchreAgent
from .othello import OthelloAgent
from .go import GoAgent
from .chess import ChessAgent
from .checkers import CheckersAgent
from .dots_and_boxes import DotsAndBoxesAgent
from .clobber import ClobberAgent
from .quoridor import QuoridorAgent
# Single-player games
from .game_2048 import Game2048Agent
from .solitaire import SolitaireAgent
# Advanced strategy games (NEW)
from .bridge import BridgeAgent
from .amazons import AmazonsAgent
from .oware import OwareAgent


# Game name -> Agent class mapping
# This registry allows looking up the appropriate agent for any game
GAME_AGENTS = {
    # Multi-player games
    "leduc_poker": LeducPokerAgent,
    "liars_dice": LiarsDiceAgent,
    "battleship": BattleshipAgent,
    "goofspiel": GoofspielAgent,
    "gin_rummy": GinRummyAgent,
    "backgammon": BackgammonAgent,
    "pig": PigAgent,
    "blackjack": BlackjackAgent,
    "phantom_ttt": PhantomTttAgent,
    "breakthrough": BreakthroughAgent,
    "hex": HexAgent,
    "hearts": HeartsAgent,
    "cribbage": CribbageAgent,
    "euchre": EuchreAgent,
    "othello": OthelloAgent,
    "go": GoAgent,
    "chess": ChessAgent,
    "checkers": CheckersAgent,
    "dots_and_boxes": DotsAndBoxesAgent,
    "clobber": ClobberAgent,
    "quoridor": QuoridorAgent,
    # Single-player games (high-quality additions)
    "2048": Game2048Agent,
    "solitaire": SolitaireAgent,
    # Advanced strategy games (NEW)
    "bridge": BridgeAgent,
    "amazons": AmazonsAgent,
    "oware": OwareAgent,
}

__all__ = ["GAME_AGENTS"]
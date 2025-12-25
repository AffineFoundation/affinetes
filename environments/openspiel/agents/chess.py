"""Chess Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class ChessAgent(BaseGameAgent):
    """Chess Game Agent"""
    
    @property
    def game_name(self) -> str:
        return "chess"
    
    def get_rules(self) -> str:
        return """CHESS RULES:
Board: 8×8 grid. Each player has 16 pieces: King, Queen, 2 Rooks, 2 Bishops, 2 Knights, 8 Pawns.
Goal: Checkmate opponent's King (King under attack with no escape).

Piece movement:
- King: 1 square any direction. Rook: Any distance horizontally/vertically.
- Bishop: Any distance diagonally. Queen: Combines Rook + Bishop.
- Knight: L-shape (2+1 squares). Pawn: 1 forward (2 on first move), captures diagonally.

Move Format: Standard algebraic notation (e.g., "e2e4", "Nf3", "O-O" for castling).

Special moves: Castling (King + Rook), En passant (pawn capture), Pawn promotion (reach 8th rank).
Winning: Checkmate opponent. Draw if stalemate, insufficient material, or repetition.

NOTE: This evaluation has a maximum turn limit to ensure completion."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Chess parameter generation
        """
        return {}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """8×8 board, 4674 actions, MaxGameLength=17695. Extremely complex game."""
        return (200, 20)

"""Battleship Game Agent - Provides game rules and state formatting for LLM evaluation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from typing import Dict, Any
from base_agent import BaseGameAgent


class BattleshipAgent(BaseGameAgent):
    """Agent for Battleship game with two-phase gameplay: ship placement and war phase"""
    
    @property
    def game_name(self) -> str:
        return "battleship"
    
    def get_rules(self) -> str:
        """Return comprehensive Battleship game rules"""
        return """BATTLESHIP RULES:

Overview: Two-player imperfect information game with two phases: ship placement and war phase.

PHASE 1: SHIP PLACEMENT
- Players alternate placing ships on their own board
- Each ship occupies consecutive cells (horizontal or vertical)
- Ships cannot overlap
- Number and size of ships determined by game configuration
- IMPORTANT: You can only see your own ships, not opponent's ships

PHASE 2: WAR PHASE
- After all ships placed, players alternate shooting at opponent's board
- Each shot targets one cell coordinate
- Three possible shot outcomes:
  - Miss (Water): Shot hit water
  - Hit: Shot hit a ship but didn't sink it
  - Sunk: Shot hit and completely destroyed a ship
- Game ends when:
  - All ships of one player are sunk
  - Both players run out of shots

COORDINATE SYSTEM:
- Board uses (row, col) coordinate system
- row: Row number, starting from 0
- col: Column number, starting from 0
- Example: (0,0) is top-left corner, (2,3) is row 3 column 4

SHIP PLACEMENT:
- Horizontal: Ship occupies cells from left to right
- Vertical: Ship occupies cells from top to bottom
- Top-left coordinate: The top-leftmost cell occupied by ship

WINNING:
- Score = value of opponent's ships sunk - value of your ships sunk
- By default all ships have equal value (zero-sum game)"""

    def format_state(self, state, player_id: int) -> str:
        """
        Format Battleship game state into human-readable format for LLM
        
        Parses information_state_string and converts into clear descriptions
        of ship placements, shot outcomes, and opponent shots.
        """
        info_state = state.information_state_string(player_id)
        
        # Extract move number
        move_match = re.search(r'T=(\d+)', info_state)
        move_number = int(move_match.group(1)) if move_match else 0
        
        # Get board dimensions from game
        game = state.get_game()
        board_width = game.get_parameters()["board_width"]
        board_height = game.get_parameters()["board_height"]
        ship_sizes_str = game.get_parameters()["ship_sizes"]
        
        # Parse ship sizes
        ship_sizes = self._parse_ship_sizes(ship_sizes_str)
        num_ships = len(ship_sizes)
        
        # Determine game phase
        # Ship placement phase: first 2*num_ships moves
        placement_phase_moves = 2 * num_ships
        
        state_parts = []
        
        # Add board info
        state_parts.append(f"Board size: {board_height}×{board_width}")
        state_parts.append(f"Ship configuration: {ship_sizes} (length list)")
        
        if move_number <= placement_phase_moves:
            # Ship Placement Phase
            state_parts.append(f"\n=== PHASE 1: SHIP PLACEMENT (Step {move_number}/{placement_phase_moves}) ===")
            
            # Parse own ship placements
            placements = re.findall(r'/([hv])_(\d+)_(\d+)', info_state)
            
            if placements:
                state_parts.append(f"\nYour placed ships ({len(placements)}/{num_ships}):")
                for i, (direction, row, col) in enumerate(placements, 1):
                    dir_str = "Horizontal" if direction == 'h' else "Vertical"
                    ship_len = ship_sizes[i-1]
                    state_parts.append(f"  Ship {i}: {dir_str}, top-left ({row},{col}), length {ship_len}")
            else:
                state_parts.append(f"\nYou haven't placed any ships yet")
            
            if len(placements) < num_ships:
                next_ship_idx = len(placements)
                next_ship_len = ship_sizes[next_ship_idx]
                state_parts.append(f"\nNext: Place ship #{next_ship_idx+1} (length {next_ship_len})")
        
        else:
            # War Phase
            war_move = move_number - placement_phase_moves
            state_parts.append(f"\n=== PHASE 2: WAR PHASE (Shot #{war_move}) ===")
            
            # Parse own ship placements (for reference)
            placements = re.findall(r'/([hv])_(\d+)_(\d+)', info_state)
            state_parts.append(f"\nYour ship layout:")
            for i, (direction, row, col) in enumerate(placements, 1):
                dir_str = "Horizontal" if direction == 'h' else "Vertical"
                ship_len = ship_sizes[i-1]
                state_parts.append(f"  Ship {i}: {dir_str}, start ({row},{col}), length {ship_len}")
            
            # Parse own shots and create grid visualization
            own_shots = re.findall(r'/shot_(\d+)_(\d+):([WHS])', info_state)
            if own_shots:
                state_parts.append(f"\nYour attack grid ({len(own_shots)} shots):")
                
                # Create attack grid (X=sunk, H=hit, M=miss, .=unknown)
                grid = [['.' for _ in range(board_width)] for _ in range(board_height)]
                recent_shots = []
                
                for row, col, outcome in own_shots:
                    r, c = int(row), int(col)
                    if outcome == 'W':
                        grid[r][c] = 'M'
                    elif outcome == 'H':
                        grid[r][c] = 'H'
                    elif outcome == 'S':
                        grid[r][c] = 'X'
                    
                    # Keep last 5 shots for history
                    if len(recent_shots) < 5:
                        recent_shots.insert(0, (r, c, outcome))
                
                # Display grid with coordinates
                state_parts.append("  " + " ".join([str(i) for i in range(board_width)]))
                for r in range(board_height):
                    state_parts.append(f"{r} " + " ".join(grid[r]))
                
                state_parts.append("\nLegend: X=SUNK, H=HIT, M=MISS, .=not shot")
                
                # Show recent shots
                if recent_shots:
                    state_parts.append(f"\nRecent shots (last {len(recent_shots)}):")
                    for r, c, outcome in reversed(recent_shots):
                        outcome_str = "SUNK" if outcome == 'S' else ("HIT" if outcome == 'H' else "MISS")
                        state_parts.append(f"  ({r},{c}): {outcome_str}")
            else:
                state_parts.append(f"\nYou haven't shot yet")
            
            # Parse opponent shots
            opp_shots = re.findall(r'/oppshot_(\d+)_(\d+)', info_state)
            if opp_shots:
                state_parts.append(f"\nOpponent's shot locations ({len(opp_shots)} shots):")
                state_parts.append("  " + ", ".join([f"({r},{c})" for r, c in opp_shots]))
                state_parts.append("  (Note: You don't know opponent's shot outcomes)")
            else:
                state_parts.append(f"\nOpponent hasn't shot yet")
        
        return "\n".join(state_parts)

    def _parse_ship_sizes(self, ship_sizes_str: str) -> list:
        """Parse ship_sizes parameter string like '[2;3;4]' into list [2,3,4]"""
        # Remove brackets and whitespace
        cleaned = ship_sizes_str.strip().strip('[]')
        # Split by semicolon
        parts = cleaned.split(';')
        # Convert to integers
        return [int(p.strip()) for p in parts if p.strip()]

    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Generate game parameters based on config_id
        
        Battleship has 3 variants with different board sizes
        """
        board_var = config_id % 3
        
        board_size = 6 + board_var * 2  # 6, 8, 10
        
        if board_var == 0:
            ship_sizes = "[2;2]"
            ship_values = "[1;1]"
        elif board_var == 1:
            ship_sizes = "[2;3;3]"
            ship_values = "[1;1;1]"
        else:
            ship_sizes = "[2;3;3;4]"
            ship_values = "[1;1;1;1]"
        
        return {
            "board_width": board_size,
            "board_height": board_size,
            "ship_sizes": ship_sizes,
            "ship_values": ship_values
        }
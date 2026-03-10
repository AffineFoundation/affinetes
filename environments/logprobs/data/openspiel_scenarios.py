"""OpenSpiel game scenarios for logprobs testing.

Each scenario contains the exact prompt format that LLMs see during OpenSpiel
game evaluation: system prompt (rules + output format) + user prompt (state +
legal actions).

Games included (matching OpenSpiel environment):
  goofspiel, leduc_poker, gin_rummy, hex, othello, clobber

Structure:
    GAMES: list of game dicts, each with:
        - name: game name (matches OpenSpiel)
        - rules: game rules text (from agent's get_rules())
        - scenarios: list of (state_desc, player_id, legal_actions) tuples
            - state_desc: formatted state string
            - player_id: 0 or 1
            - legal_actions: list of "id -> action_str" strings
"""

# -- System prompt template (matches BaseGameAgent.generate_system_prompt) --
SYSTEM_TEMPLATE = """\
You are playing {game_name}.

# Game Rules
{rules}

# Output Format
You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.

Examples:
- For action "0 -> roll": respond "0"
- For action "89 -> a3": respond "89"\
"""

# -- User prompt template (matches BaseGameAgent.generate_user_prompt) --
USER_TEMPLATE = """\
Current State:
{state_desc}

You are Player {player_id}.
Legal Actions:
{legal_actions}

Your choice (ID only):\
"""


GAMES = [
    # ================================================================
    # goofspiel - Bidding strategy game
    # ================================================================
    {
        "name": "goofspiel",
        "rules": """GOOFSPIEL RULES:
Setup: Each player has bid cards numbered 1 to N. A prize deck with cards 1 to N is shuffled.
Goal: Win the most points by bidding on prize cards.

Each turn:
1. Reveal top prize card (worth its face value in points)
2. Players simultaneously play one bid card from their hand
3. Highest bidder wins the prize card (adds its value to score)
4. If bids tie, prize card is discarded (no one gets points)

Winning: Player with most points after all rounds wins.""",
        "scenarios": [
            # Early game, full hand
            (
                "Current point card: 6\n"
                "Player 0: 0 points, Player 1: 0 points\n"
                "My remaining bid cards: 1 2 3 4 5 6 7 8\n"
                "Win sequence: \n"
                "(Win sequence: 1=player 1 won, 0=player 0 won, negative=tie)",
                0,
                ["0 -> 1", "1 -> 2", "2 -> 3", "3 -> 4", "4 -> 5", "5 -> 6", "6 -> 7", "7 -> 8"],
            ),
            # Mid game, some cards played
            (
                "Current point card: 3\n"
                "Player 0: 14 points, Player 1: 4 points\n"
                "My remaining bid cards: 1 3 5 7\n"
                "Win sequence: 1 0 0 -3 1\n"
                "(Win sequence: 1=player 1 won, 0=player 0 won, negative=tie)",
                0,
                ["0 -> 1", "2 -> 3", "4 -> 5", "6 -> 7"],
            ),
            # Late game, few cards left, close score
            (
                "Current point card: 8\n"
                "Player 0: 18 points, Player 1: 20 points\n"
                "My remaining bid cards: 2 6\n"
                "Win sequence: 1 0 1 0 0 1\n"
                "(Win sequence: 1=player 1 won, 0=player 0 won, negative=tie)",
                1,
                ["1 -> 2", "5 -> 6"],
            ),
            # High prize card, large hand
            (
                "Current point card: 10\n"
                "Player 0: 5 points, Player 1: 8 points\n"
                "My remaining bid cards: 2 4 6 8 10\n"
                "Win sequence: 0 1\n"
                "(Win sequence: 1=player 1 won, 0=player 0 won, negative=tie)",
                1,
                ["1 -> 2", "3 -> 4", "5 -> 6", "7 -> 8", "9 -> 10"],
            ),
        ],
    },
    # ================================================================
    # leduc_poker - Poker reasoning
    # ================================================================
    {
        "name": "leduc_poker",
        "rules": """LEDUC POKER RULES:

Deck: 2 suits \u00d7 (num_players + 1) ranks. For 2 players: 6 cards (J\u2660 J\u2665 Q\u2660 Q\u2665 K\u2660 K\u2665).

Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.

Round 1: Each player receives one private card. Actions: Fold (lose ante), Call/Check (match current bet), Raise (add 2 chips to bet). Maximum 2 raises per round.
Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.

Winning: Player with best hand wins pot (or last remaining if others fold).
Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).""",
        "scenarios": [
            # Round 1, first to act with King
            (
                "Your card: K\u2660\n"
                "Current round: 1/2\n"
                "Pot size: 2 chips\n"
                "Your chips: 99\n"
                "Opponent chips: 99\n"
                "Round 1 betting: (no actions yet)\n"
                "Your turn to act",
                0,
                ["0 -> Fold", "1 -> Call", "2 -> Raise"],
            ),
            # Round 2, public card revealed, have a pair
            (
                "Your card: Q\u2665\n"
                "Public card: Q\u2660\n"
                "Hand: Pair\n"
                "Current round: 2/2\n"
                "Pot size: 6 chips\n"
                "Your chips: 97\n"
                "Opponent chips: 97\n"
                "Round 1 betting: Call, Raise, Call\n"
                "Round 2 betting: (no actions yet)\n"
                "Your turn to act",
                0,
                ["0 -> Fold", "1 -> Call", "2 -> Raise"],
            ),
            # Facing a raise with Jack
            (
                "Your card: J\u2660\n"
                "Current round: 1/2\n"
                "Pot size: 4 chips\n"
                "Your chips: 99\n"
                "Opponent chips: 97\n"
                "Round 1 betting: Raise\n"
                "Your turn to act",
                1,
                ["0 -> Fold", "1 -> Call", "2 -> Raise"],
            ),
            # Round 2, no pair, high card K vs public J
            (
                "Your card: K\u2665\n"
                "Public card: J\u2660\n"
                "Current round: 2/2\n"
                "Pot size: 4 chips\n"
                "Your chips: 98\n"
                "Opponent chips: 98\n"
                "Round 1 betting: Call, Call\n"
                "Round 2 betting: Raise\n"
                "Your turn to act",
                1,
                ["0 -> Fold", "1 -> Call", "2 -> Raise"],
            ),
        ],
    },
    # ================================================================
    # gin_rummy - Card strategy
    # ================================================================
    {
        "name": "gin_rummy",
        "rules": """GIN RUMMY RULES:

SETUP:
- 52-card deck, each player receives 7-10 cards (variant dependent)
- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)

MELDS (Valid Combinations):
1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)
2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)
Examples:
- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663
- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)

CARD NOTATION:
- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)
- Suits: s(spades\u2660), h(hearts\u2665), d(diamonds\u2666), c(clubs\u2663)
- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades

GAME PHASES:
1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)
2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)
3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)
4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)
5. Knock: Declare end of hand when deadwood \u2264 knock_card value

EACH TURN:
1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)
2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)

KNOCKING:
- When deadwood \u2264 knock_card value (8-10), you MAY knock to end hand
- Gin: ALL cards form melds (0 deadwood) = 25-point bonus

SCORING: Winner scores difference in deadwood point values.
Card Values: A=1, 2-10=face value, J=11, Q=12, K=13

IMPORTANT: Always respond with the action ID number ONLY, never card names.""",
        "scenarios": [
            # FirstUpcard phase - decide whether to pick up revealed card
            (
                "Hand: As 3s 4s 7h 8h 9h Jd\n"
                "Upcard: 5s\n"
                "Stock: 44 cards remaining\n"
                "Phase: FirstUpcard",
                0,
                ["52 -> Draw upcard", "54 -> Pass"],
            ),
            # Draw phase - pick from upcard or stock
            (
                "Hand: As 3s 4s 5s 7h 8h 9h Jd\n"
                "Upcard: Qc\n"
                "Stock: 38 cards remaining\n"
                "Phase: Draw\n"
                "Opponent drew from stock last turn",
                0,
                ["52 -> Draw upcard", "53 -> Draw stock"],
            ),
            # Discard phase - choose card to discard from hand
            (
                "Hand: As 3s 4s 5s 7h 8h 9h Jd Qc\n"
                "Melds found: [3s-4s-5s] [7h-8h-9h]\n"
                "Deadwood: As(1) + Jd(11) + Qc(12) = 24\n"
                "Phase: Discard",
                0,
                [
                    "0 -> As", "2 -> 3s", "3 -> 4s", "4 -> 5s",
                    "19 -> 7h", "20 -> 8h", "21 -> 9h",
                    "36 -> Jd", "50 -> Qc",
                ],
            ),
            # Discard phase with knock option - low deadwood
            (
                "Hand: 2c 3c 4c 5c Ts Th Td Kh\n"
                "Melds found: [2c-3c-4c-5c] [Ts-Th-Td]\n"
                "Deadwood: Kh(13) = 13\n"
                "Knock card value: 10\n"
                "Phase: Discard",
                1,
                [
                    "40 -> 2c", "41 -> 3c", "42 -> 4c", "43 -> 5c",
                    "9 -> Ts", "22 -> Th", "35 -> Td",
                    "25 -> Kh",
                ],
            ),
            # Draw phase mid-game, opponent discarded useful card
            (
                "Hand: 6d 7d 9d Jh Qh Kh 2c\n"
                "Upcard: 8d\n"
                "Stock: 30 cards remaining\n"
                "Phase: Draw\n"
                "Opponent discarded 8d",
                1,
                ["52 -> Draw upcard", "53 -> Draw stock"],
            ),
            # Layoff phase - add cards to opponent's melds after they knock
            (
                "Hand: 5s 9c Kd\n"
                "Opponent knocked with melds: [5h-5d-5c] [9s-Ts-Js]\n"
                "Phase: Layoff",
                0,
                ["4 -> 5s", "47 -> 9c", "54 -> Pass"],
            ),
        ],
    },
    # ================================================================
    # hex - Path planning
    # ================================================================
    {
        "name": "hex",
        "rules": """HEX RULES:
Board: Diamond-shaped grid (5\u00d75, 7\u00d77, 9\u00d79, or 11\u00d711). Two players (Red and Blue).
Goal: Connect your two opposite sides of the board with an unbroken chain of your stones.

Turn: Place one stone of your color on any empty cell.
Red (x) connects top-left to bottom-right sides.
Blue (o) connects top-right to bottom-left sides.

No draws possible: Someone must win.""",
        "scenarios": [
            # Early 5x5 game
            (
                "Board (5x5):\n"
                "  a b c d e\n"
                "1  . . . . .\n"
                "2   . . . . .\n"
                "3    . . x . .\n"
                "4     . . . . .\n"
                "5      . . . . .",
                0,
                [
                    "0 -> a1", "1 -> b1", "2 -> c1", "3 -> d1", "4 -> e1",
                    "5 -> a2", "6 -> b2", "7 -> c2", "8 -> d2", "9 -> e2",
                    "10 -> a3", "11 -> b3", "13 -> d3", "14 -> e3",
                    "15 -> a4", "16 -> b4", "17 -> c4", "18 -> d4", "19 -> e4",
                    "20 -> a5", "21 -> b5", "22 -> c5", "23 -> d5", "24 -> e5",
                ],
            ),
            # Mid game 5x5 with several stones placed
            (
                "Board (5x5):\n"
                "  a b c d e\n"
                "1  . x . . .\n"
                "2   . . o . .\n"
                "3    . x x o .\n"
                "4     . o . . .\n"
                "5      . . . . .",
                0,
                [
                    "0 -> a1", "2 -> c1", "3 -> d1", "4 -> e1",
                    "5 -> a2", "6 -> b2", "8 -> d2", "9 -> e2",
                    "10 -> a3", "14 -> e3",
                    "15 -> a4", "17 -> c4", "18 -> d4", "19 -> e4",
                    "20 -> a5", "21 -> b5", "22 -> c5", "23 -> d5", "24 -> e5",
                ],
            ),
            # 7x7 board early game
            (
                "Board (7x7):\n"
                "  a b c d e f g\n"
                "1  . . . . . . .\n"
                "2   . . . . . . .\n"
                "3    . . . . . . .\n"
                "4     . . . x . . .\n"
                "5      . . . . . . .\n"
                "6       . . . . . . .\n"
                "7        . . . . . . .",
                1,
                [
                    "0 -> a1", "1 -> b1", "2 -> c1", "3 -> d1", "4 -> e1", "5 -> f1", "6 -> g1",
                    "7 -> a2", "8 -> b2", "9 -> c2", "10 -> d2", "11 -> e2", "12 -> f2", "13 -> g2",
                    "14 -> a3", "15 -> b3", "16 -> c3", "17 -> d3", "18 -> e3", "19 -> f3", "20 -> g3",
                    "21 -> a4", "22 -> b4", "23 -> c4", "25 -> e4", "26 -> f4", "27 -> g4",
                    "28 -> a5", "29 -> b5", "30 -> c5", "31 -> d5", "32 -> e5", "33 -> f5", "34 -> g5",
                    "35 -> a6", "36 -> b6", "37 -> c6", "38 -> d6", "39 -> e6", "40 -> f6", "41 -> g6",
                    "42 -> a7", "43 -> b7", "44 -> c7", "45 -> d7", "46 -> e7", "47 -> f7", "48 -> g7",
                ],
            ),
        ],
    },
    # ================================================================
    # othello - Spatial reasoning
    # ================================================================
    {
        "name": "othello",
        "rules": """OTHELLO (REVERSI) RULES:
Board: 8\u00d78 grid. 2 players (Black and White). Start with 4 discs in center (2 black, 2 white diagonal).
Goal: Have more discs of your color when board is full or no moves available.

Turn: Place disc to sandwich opponent's discs between your new disc and existing disc (horizontally, vertically, or diagonally). All sandwiched opponent discs flip to your color.
Must flip at least 1 disc; if no valid move, pass turn.

Winning: Player with most discs when game ends wins.""",
        "scenarios": [
            # Opening position
            (
                "Board:\n"
                "  a b c d e f g h\n"
                "1 . . . . . . . .\n"
                "2 . . . . . . . .\n"
                "3 . . . . . . . .\n"
                "4 . . . o x . . .\n"
                "5 . . . x o . . .\n"
                "6 . . . . . . . .\n"
                "7 . . . . . . . .\n"
                "8 . . . . . . . .\n"
                "Black(x): 2  White(o): 2",
                0,
                ["19 -> d3", "26 -> c4", "37 -> f5", "44 -> e6"],
            ),
            # Mid game position
            (
                "Board:\n"
                "  a b c d e f g h\n"
                "1 . . . . . . . .\n"
                "2 . . . . . . . .\n"
                "3 . . x x x . . .\n"
                "4 . . x o x . . .\n"
                "5 . . . x o o . .\n"
                "6 . . . . o . . .\n"
                "7 . . . . . . . .\n"
                "8 . . . . . . . .\n"
                "Black(x): 7  White(o): 4",
                1,
                ["17 -> b3", "33 -> b5", "40 -> a6", "41 -> b6", "42 -> c6", "47 -> h6", "50 -> c7"],
            ),
            # Late game, many pieces on board
            (
                "Board:\n"
                "  a b c d e f g h\n"
                "1 . . . . . . . .\n"
                "2 . . o o o . . .\n"
                "3 . o o x o o . .\n"
                "4 . o x x x o . .\n"
                "5 . . x x x o . .\n"
                "6 . . x o x . . .\n"
                "7 . . . x . . . .\n"
                "8 . . . . . . . .\n"
                "Black(x): 11  White(o): 11",
                0,
                [
                    "0 -> a1", "8 -> a2", "13 -> f2",
                    "16 -> a3", "22 -> g3",
                    "24 -> a4", "31 -> h4",
                    "32 -> a5", "39 -> h5",
                    "40 -> a6", "45 -> f6", "46 -> g6",
                    "50 -> c7", "52 -> e7",
                ],
            ),
        ],
    },
    # ================================================================
    # clobber - Capture tactics
    # ================================================================
    {
        "name": "clobber",
        "rules": """CLOBBER RULES:
Board: Rectangular grid with alternating black (x) and white (o) pieces.
Goal: Make the last capture. A player who cannot capture loses.

Turn: Move one of your pieces to an adjacent cell (up/down/left/right) that contains an opponent's piece. Your piece replaces (captures) the opponent's piece.
You CANNOT move to an empty cell or a cell with your own piece.

Winning: The player who makes the last capture wins. If you cannot move, you lose.""",
        "scenarios": [
            # 4x5 board opening
            (
                "Board (4x5):\n"
                "x o x o\n"
                "o x o x\n"
                "x o x o\n"
                "o x o x\n"
                "x o x o",
                0,
                [
                    "0 -> a1-b1", "2 -> c1-b1", "2 -> c1-d1",
                    "5 -> b2-a2", "5 -> b2-c2", "7 -> d2-c2",
                    "8 -> a3-b3", "10 -> c3-b3", "10 -> c3-d3",
                    "13 -> b4-a4", "13 -> b4-c4", "15 -> d4-c4",
                    "16 -> a5-b5", "18 -> c5-b5", "18 -> c5-d5",
                ],
            ),
            # Mid game with some captures done
            (
                "Board (4x5):\n"
                "x . x o\n"
                "o x o x\n"
                "x o . o\n"
                "o x o x\n"
                ". o x o",
                0,
                [
                    "0 -> a1-a2", "2 -> c1-d1",
                    "5 -> b2-a2", "5 -> b2-c2", "7 -> d2-c2",
                    "8 -> a3-b3",
                    "13 -> b4-a4", "13 -> b4-c4", "15 -> d4-c4",
                    "18 -> c5-b5", "18 -> c5-d5",
                ],
            ),
            # Late game, few pieces remain
            (
                "Board (4x5):\n"
                ". . . .\n"
                ". x o .\n"
                ". . . .\n"
                "o x . .\n"
                ". . . o",
                0,
                [
                    "5 -> b2-c2",
                    "13 -> b4-a4",
                ],
            ),
        ],
    },
]

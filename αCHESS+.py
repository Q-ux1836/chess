import sys
import os
import subprocess
import time
from datetime import datetime # Import datetime for the clock

# Set a higher recursion limit for the AI's negamax algorithm
# This is often necessary for recursive algorithms in Python to prevent RecursionError
sys.setrecursionlimit(50000000) # Increased limit for deep recursion

from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
                            QGraphicsPixmapItem, QGraphicsRectItem, QVBoxLayout, QWidget,
                            QPushButton, QHBoxLayout, QGraphicsEllipseItem, QLabel,
                            QMessageBox, QDialog, QGridLayout, QFrame, QSizePolicy, QGraphicsTextItem,
                            QMenu, QColorDialog, QSlider, QAction) # Added QAction for menu items
from PyQt5.QtGui import QPixmap, QColor, QBrush, QPainter, QIcon, QFont, QPen, QLinearGradient
from PyQt5.QtCore import QRectF, Qt, QTimer, QPointF, QSize, QUrl, pyqtSlot, QPoint # Import QPoint
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent # Added QMediaPlayer, QMediaContent

# Conditional import for QtWebEngineWidgets, as it might not be installed initially
WEBENGINE_AVAILABLE = False
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
    from PyQt5.QtWebChannel import QWebChannel # Explicitly import QWebChannel
    WEBENGINE_AVAILABLE = True
except ImportError:
    print("PyQtWebEngine is not found. Homepage and custom HTML popup features will be disabled.")


# --- Dependency Installation (Attempts to install missing modules) ---
# It's generally better to manage dependencies via a requirements.txt file
# and instruct the user to run 'pip install -r requirements.txt' manually.
# However, if you absolutely need to attempt an automatic installation,
# you can use the subprocess module. This might require administrative privileges
# and can lead to unexpected behavior or environment issues.

def install_module(module_name: str, package_name: str = None):
    """
    Installs a Python module using pip.
    Args:
        module_name (str): The name of the module to try importing (e.g., 'PyQt5', 'PyQt5.QtWebEngineWidgets').
        package_name (str, optional): The name of the package to install via pip
                                      if different from module_name (e.g., 'PyQtWebEngine' for 'PyQt5.QtWebEngineWidgets').
                                      Defaults to module_name if not provided.
    """
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"'{module_name}' is already installed.")
        return True
    except ImportError:
        print(f"'{module_name}' not found. Attempting to install '{package_name}'...")
        try:
            # Use sys.executable to ensure pip is run for the current Python interpreter
            # check=True will raise CalledProcessError if the command fails
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                check=True,
                capture_output=True, # Capture stdout and stderr
                text=True # Decode output as text
            )
            print(f"Successfully installed '{package_name}':\n{result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing '{package_name}':\n{e.stderr}")
            print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
            return False
        except FileNotFoundError:
            print(f"Error: '{sys.executable}' (Python interpreter) not found. Make sure Python is in your PATH.")
            print(f"Please install '{package_name}' manually using: pip install {package_name}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during installation of '{package_name}': {e}")
            print(f"Please install '{package_name}' manually using: pip install {package_name}")
            return False

# List of essential modules and their corresponding package names if different
required_modules = [
    ("PyQt5", "PyQt5"),
    ("PyQt5.QtWebEngineWidgets", "PyQtWebEngine"), # This is for the homepage and custom popup
    ("PyQt5.QtMultimedia", "PyQt5-QtMultimedia") # For sound playback
]

temp_webengine_available_check = True
for module, package in required_modules:
    if not install_module(module, package):
        temp_webengine_available_check = False
        # If PyQtWebEngine fails to install, explicitly set WEBENGINE_AVAILABLE to False
        if module == "PyQt5.QtWebEngineWidgets":
            WEBENGINE_AVAILABLE = False

if not temp_webengine_available_check:
    print("\nWarning: Some dependencies could not be installed automatically.")
    print("The application may run with limited features (e.g., no interactive homepage).")
    # If PyQt5 itself failed, we should exit.
    if not install_module("PyQt5", "PyQt5"): # Re-check PyQt5 specifically
        print("Fatal: PyQt5 is not installed. Exiting application.")
        sys.exit(1)
else:
    # If all installations were attempted and succeeded, and initial import worked, then set True
    WEBENGINE_AVAILABLE = True


# --- Helper Functions (can be moved to a separate file for larger projects) ---
def download_image_if_not_exists(filename, subdirectory="."):
    """
    Checks if an image exists locally.
    Creates the subdirectory if it doesn't exist.
    Returns the local path to the image, or None if it doesn't exist.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, subdirectory)
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, filename)

    if os.path.exists(local_path):
        return local_path
    else:
        print(f"Local file {filename} not found at {local_path}.")
        return None

def get_local_file_url(filename, subdirectory="."):
    """
    Returns a QUrl for a local file, creating the subdirectory if it doesn't exist.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, subdirectory)
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, filename)
    return QUrl.fromLocalFile(local_path)


# --- Chess Game Logic Class ---
class ChessLogic:
    """
    Manages the core chess game logic, including board state, piece movements,
    check/checkmate/stalemate detection, and AI decision-making.
    """
    def __init__(self):
        """
        Initializes the chess board and game state variables.
        """
        self.board = self.create_initial_board()
        self.current_turn = 'w'  # White starts first
        self.game_over = False
        self.winner: str | None = None
        # move_history stores detailed info for undo functionality and AI simulation
        # Format: (start_pos, end_pos, moved_piece, captured_piece, promoted_to, castling_info, en_passant_target)
        self.move_history = []
        self.redo_history = [] # New: Stores undone moves for redo functionality
        self.captured_pieces = {'w': [], 'b': []}
        self.kings_moved = {'w': False, 'b': False} # Tracks if king has moved for castling
        self.rooks_moved = { # Tracks if rooks have moved for castling
            'w': {'kingside': False, 'queenside': False},
            'b': {'kingside': False, 'queenside': False}
        }
        self.en_passant_target = None # Stores (row, col) of square behind pawn that just double-moved

        # Piece values for AI evaluation (standard values, multiplied for integer arithmetic)
        self.piece_values = {
            'P': 1000, 'N': 30000, 'B': 30000, 'R': 50001, 'Q': 900001, 'K': 900000,
            'p': -1000, 'n': -30000, 'b': -30000, 'r': -50001, 'q': -900001, 'k': -900000
        }
        # Positional piece values (example, can be expanded for more sophisticated AI)
        # These tables penalize pawns on the back rank, encourage knights in center, etc.
        self.pawn_table = [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ]
        self.knight_table = [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ]
        self.bishop_table = [
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ]
        self.rook_table = [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [0,  0,  0,  5,  5,  0,  0,  0]
        ]
        self.queen_table = [
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [-5,  0,  5,  5,  5,  5,  0, -5],
            [0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ]
        self.king_table_midgame = [
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [20, 20,  0,  0,  0,  0, 20, 20],
            [20, 30, 10,  0,  0, 10, 30, 20]
        ]
        self.king_table_endgame = [
            [-50,-40,-30,-20,-20,-30,-40,-50],
            [-30,-20,-10,  0,  0,-10,-20,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-30,  0,  0,  0,  0,-30,-30],
            [-50,-30,-30,-30,-30,-30,-30,-50]
        ]

        self.ai_difficulty = 'easy' # New: AI difficulty setting

    def create_initial_board(self) -> list[list[str | None]]:
        """
        Creates the standard 8x8 chess board and places pieces in their
        initial starting positions.
        """
        board: list[list[str | None]] = [[None for _ in range(8)] for _ in range(8)]

        # White pieces (row 6 and 7)
        board[7] = ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
        board[6] = ['wP'] * 8

        # Black pieces (row 0 and 1)
        board[0] = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR']
        board[1] = ['bP'] * 8

        return board

    def get_piece(self, row: int, col: int) -> str | None:
        """
        Returns the piece string ('wP', 'bK', etc.) at the specified board position.
        Returns None if the position is out of bounds or empty.
        """
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None

    def place_piece(self, row: int, col: int, piece: str | None):
        """
        Places a piece (or None to clear) at the specified board position.
        """
        if 0 <= row < 8 and 0 <= col < 8:
            self.board[row][col] = piece

    def find_king(self, color: str) -> tuple[int, int] | None:
        """
        Finds and returns the (row, col) position of the king of the specified color.
        Returns None if the king is not found (should not happen in a valid game).
        """
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                if piece and piece[0] == color and piece[1] == 'K':
                    return (r, c)
        return None

    def is_square_attacked(self, king_color: str, square_pos: tuple[int, int]) -> bool:
        """
        Checks if a given square is attacked by any of the opponent's pieces.
        This is crucial for king safety, castling, and check detection.
        """
        opponent_color = 'b' if king_color == 'w' else 'w'
        target_row, target_col = square_pos

        # Iterate through all squares on the board
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                # If it's an opponent's piece
                if piece and piece[0] == opponent_color:
                    # Temporarily change current_turn to check opponent's moves
                    # This is important because highlight_moves uses current_turn
                    original_turn = self.current_turn
                    self.current_turn = opponent_color

                    # Get moves for the opponent's piece without checking for king safety
                    # because we are checking if a square is attacked, not if the move is legal for the opponent.
                    moves = self.highlight_moves((r, c), check_for_check=False)
                    self.current_turn = original_turn # Revert turn immediately

                    # If the target square is in the opponent's possible moves, it's attacked
                    if square_pos in moves:
                        return True
        return False

    def is_opponent_in_check(self, color: str) -> bool:
        """
        Checks if the king of the specified color is currently in check.
        This is done by checking if the king's current square is attacked by the opponent.
        """
        king_pos = self.find_king(color)
        if not king_pos:
            return False # Should not happen in a normal game

        # Check if the king's square is attacked by the opponent's pieces
        return self.is_square_attacked(color, king_pos)

    def is_checkmate(self, color: str) -> bool:
        """
        Determines if the player of the given color is in checkmate.
        A player is in checkmate if their king is in check AND they have no legal moves
        to get out of check.
        """
        # First, check if the king is actually in check
        if not self.is_opponent_in_check(color):
            return False # Not in check, so cannot be checkmate

        # Iterate through all pieces of the current player
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                if piece and piece[0] == color:
                    # Get all legal moves for this piece.
                    # highlight_moves with check_for_check=True already filters out moves
                    # that would leave the king in check.
                    valid_moves = self.highlight_moves((r, c), check_for_check=True)
                    if valid_moves:
                        return False # Found at least one legal move, so not checkmate

        # No legal moves found for any piece, and king is in check, so it's checkmate
        return True

    def is_stalemate(self, color: str) -> bool:
        """
        Determines if the player of the given color is in stalemate.
        A player is in stalemate if their king is NOT in check AND they have no legal moves.
        """
        # First, check if the king is NOT in check
        if self.is_opponent_in_check(color):
            return False # In check, so cannot be stalemate (could be checkmate)

        # Iterate through all pieces of the current player
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                if piece and piece[0] == color:
                    # Get all legal moves for this piece.
                    # highlight_moves with check_for_check=True filters out moves
                    # that would put the king in check.
                    valid_moves = self.highlight_moves((r, c), check_for_check=True)
                    if valid_moves:
                        return False # Found at least one legal move, so not stalemate

        # No legal moves found for any piece, and king is not in check, so it's stalemate
        return True

    def highlight_moves(self, selected_pos: tuple[int, int], check_for_check: bool = True) -> list[tuple[int, int]]:
        """
        Calculates and returns a list of valid (row, col) positions a piece at `selected_pos`
        can move to.

        Args:
            selected_pos (tuple[int, int]): The (row, col) of the piece to move.
            check_for_check (bool): If True, filters out moves that would leave the king in check.
                                    Set to False for internal checks (e.g., is_square_attacked).

        Returns:
            list[tuple[int, int]]: A list of valid destination (row, col) tuples.
        """
        row, col = selected_pos
        piece = self.get_piece(row, col)
        if piece is None:
            return []

        piece_color = piece[0]  # 'w' for white, 'b' for black
        valid_moves = []

        # Dispatch to specific piece movement logic
        if piece[1] == 'P':
            valid_moves = self.highlight_pawn_moves(row, col, piece_color)
        elif piece[1] == 'R':
            valid_moves = self.highlight_rook_moves(row, col, piece_color)
        elif piece[1] == 'N':
            valid_moves = self.highlight_knight_moves(row, col, piece_color)
        elif piece[1] == 'B':
            valid_moves = self.highlight_bishop_moves(row, col, piece_color)
        elif piece[1] == 'Q':
            valid_moves = self.highlight_queen_moves(row, col, piece_color)
        elif piece[1] == 'K':
            valid_moves = self.highlight_king_moves(row, col, piece_color)

        # Filter out moves that would put the king in check (if check_for_check is True)
        if check_for_check:
            legal_moves = []
            for move in valid_moves:
                # Simulate the move to check for king safety
                temp_board = [row[:] for row in self.board] # Create a deep copy of the board

                # Perform the simulated move
                dest_row, dest_col = move
                # Store info about captured piece for undoing during simulation
                original_piece_at_dest = temp_board[dest_row][dest_col]
                temp_board[dest_row][dest_col] = temp_board[row][col] # Move piece
                temp_board[row][col] = None # Clear original square

                # Special handling for en passant capture during simulation
                is_en_passant_sim = False
                captured_ep_pawn_sim = None
                if piece[1] == 'P' and move == self.en_passant_target:
                    # En passant captures the pawn behind the target square
                    # Use `selected_pos` for pawn's original row
                    captured_pawn_row = selected_pos[0]
                    captured_pawn_col = move[1] # End column is the captured pawn's column
                    captured_ep_pawn_sim = temp_board[captured_pawn_row][captured_pawn_col]
                    temp_board[captured_pawn_row][captured_pawn_col] = None
                    is_en_passant_sim = True

                # Special handling for castling during simulation
                if piece[1] == 'K' and abs(col - dest_col) == 2:
                    king_row = row
                    if dest_col > col: # Kingside castling
                        rook_start_col = 7
                        rook_end_col = dest_col - 1
                    else: # Queenside castling
                        rook_start_col = 0
                        rook_end_col = dest_col + 1
                    rook_piece_sim = temp_board[king_row][rook_start_col]
                    temp_board[king_row][rook_end_col] = rook_piece_sim
                    temp_board[king_row][rook_start_col] = None


                # Temporarily swap the board to the simulated state for check detection
                original_board = self.board
                self.board = temp_board

                # Check if the king is in check after the simulated move
                king_in_check_after_move = self.is_opponent_in_check(piece_color) # Check if *own* king is in check

                # Revert the board to its original state
                self.board = original_board

                # Revert the temporary changes made for this simulation for king/rook moved flags
                # This is crucial because `is_square_attacked` relies on a clean state for opponent moves.
                # If these flags were changed *during* a simulation in is_square_attacked, it might affect subsequent `highlight_moves` calls.
                # However, the logic in _update_game_state_after_move (which is called by negamax) already handles this for the main simulation.
                # The `is_square_attacked` function itself does NOT modify `kings_moved` or `rooks_moved`, so no explicit revert needed here.


                # If the king is not in check after the move, it's a legal move
                if not king_in_check_after_move:
                    legal_moves.append(move)

            return legal_moves

        return valid_moves

    def highlight_pawn_moves(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        """
        Calculates valid moves for a pawn at (row, col) of a given color.
        Includes single/double moves, captures, and en passant.
        """
        valid_moves = []
        direction = -1 if color == 'w' else 1  # White pawns move up (-1 row), black pawns move down (+1 row)
        start_row = 6 if color == 'w' else 1  # Starting row for pawns to allow double move

        # 1. Normal move (1 square forward)
        target_row_1 = row + direction
        if 0 <= target_row_1 < 8 and self.get_piece(target_row_1, col) is None:
            valid_moves.append((target_row_1, col))

            # 2. Double move from starting position
            target_row_2 = row + 2 * direction
            if row == start_row and self.get_piece(target_row_2, col) is None:
                valid_moves.append((target_row_2, col))

        # 3. Capture moves (diagonally)
        for dc in [-1, 1]: # Check left and right diagonals
            target_row_cap = row + direction
            target_col_cap = col + dc
            if 0 <= target_row_cap < 8 and 0 <= target_col_cap < 8:
                piece_to_capture = self.get_piece(target_row_cap, target_col_cap)
                if piece_to_capture and piece_to_capture[0] != color:
                    valid_moves.append((target_row_cap, target_col_cap))

        # 4. En Passant
        # Check if there's an en passant target square set from the opponent's last move
        if self.en_passant_target:
            ep_row, ep_col = self.en_passant_target
            # An en passant capture is only valid if the pawn is on the 5th rank (for white) or 4th rank (for black)
            # and the target square is diagonally adjacent to the pawn's current position.
            if (color == 'w' and row == 3 and ep_row == 2 and abs(col - ep_col) == 1) or \
               (color == 'b' and row == 4 and ep_row == 5 and abs(col - ep_col) == 1):
                valid_moves.append(self.en_passant_target)

        return valid_moves

    def highlight_rook_moves(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        """
        Calculates valid moves for a rook at (row, col) of a given color.
        Rooks move horizontally and vertically any number of squares.
        """
        valid_moves = []

        # Define directions: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row, col
            while True:
                r += dr
                c += dc
                # Check if the new position is within board bounds
                if not (0 <= r < 8 and 0 <= c < 8):
                    break # Out of bounds, stop in this direction

                piece = self.get_piece(r, c)
                if piece is None:
                    valid_moves.append((r, c)) # Empty square, can move here
                elif piece[0] != color:
                    valid_moves.append((r, c)) # Opponent's piece, can capture and stop
                    break
                else:
                    break # Own piece, cannot move through or capture, stop
        return valid_moves

    def highlight_knight_moves(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        """
        Calculates valid moves for a knight at (row, col) of a given color.
        Knights move in an 'L' shape.
        """
        valid_moves = []
        # All 8 possible 'L' shaped moves for a knight
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for dr, dc in knight_moves:
            r, c = row + dr, col + dc
            # Check if the new position is within board bounds
            if 0 <= r < 8 and 0 <= c < 8:
                piece = self.get_piece(r, c)
                # Can move if square is empty or contains an opponent's piece
                if piece is None or piece[0] != color:
                    valid_moves.append((r, c))
        return valid_moves

    def highlight_bishop_moves(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        """
        Calculates valid moves for a bishop at (row, col) of a given color.
        Bishops move diagonally any number of squares.
        """
        valid_moves = []

        # Define diagonal directions: up-left, up-right, down-left, down-right
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = row, col
            while True:
                r += dr
                c += dc
                # Check if the new position is within board bounds
                if not (0 <= r < 8 and 0 <= c < 8):
                    break # Out of bounds, stop in this direction

                piece = self.get_piece(r, c)
                if piece is None:
                    valid_moves.append((r, c)) # Empty square, can move here
                elif piece[0] != color:
                    valid_moves.append((r, c)) # Opponent's piece, can capture and stop
                    break
                else:
                    break # Own piece, cannot move through or capture, stop
        return valid_moves

    def highlight_queen_moves(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        """
        Calculates valid moves for a queen at (row, col) of a given color.
        Queens combine the moves of a rook and a bishop.
        """
        # Queen moves are simply the union of rook moves and bishop moves
        return self.highlight_rook_moves(row, col, color) + self.highlight_bishop_moves(row, col, color)

    def highlight_king_moves(self, row: int, col: int, color: str) -> list[tuple[int, int]]:
        """
        Calculates valid moves for a king at (row, col) of a given color.
        Kings move one square in any direction, and can castle.
        """
        valid_moves = []

        # King moves one square in any of the 8 directions
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                piece = self.get_piece(r, c)
                if piece is None or piece[0] != color:
                    valid_moves.append((r, c))

        # Castling logic
        # Castling is only possible if:
        # 1. The king has not moved.
        # 2. The relevant rook has not moved.
        # 3. There are no pieces between the king and the rook.
        # 4. The king is not currently in check.
        # 5. The king does not pass through or land on a square attacked by an opponent's piece.
        if not self.kings_moved[color]:
            king_row = 7 if color == 'w' else 0

            # Kingside castling (short castling)
            # King moves from e1/e8 to g1/g8, Rook moves from h1/h8 to f1/f8
            if not self.rooks_moved[color]['kingside']:
                # Check if squares f1/f8 and g1/g8 are empty
                if self.get_piece(king_row, 5) is None and self.get_piece(king_row, 6) is None:
                    # Check if king is not in check, and does not pass through or land on attacked squares
                    if not self.is_square_attacked(color, (king_row, 4)) and \
                       not self.is_square_attacked(color, (king_row, 5)) and \
                       not self.is_square_attacked(color, (king_row, 6)):
                        valid_moves.append((king_row, 6))  # Add the kingside castling move (king's destination)

            # Queenside castling (long castling)
            # King moves from e1/e8 to c1/c8, Rook moves from a1/a8 to d1/d8
            if not self.rooks_moved[color]['queenside']:
                # Check if squares b1/b8, c1/c8, and d1/d8 are empty
                if self.get_piece(king_row, 1) is None and \
                   self.get_piece(king_row, 2) is None and \
                   self.get_piece(king_row, 3) is None:
                    # Check if king is not in check, and does not pass through or land on attacked squares
                    if not self.is_square_attacked(color, (king_row, 4)) and \
                       not self.is_square_attacked(color, (king_row, 3)) and \
                       not self.is_square_attacked(color, (king_row, 2)):
                        valid_moves.append((king_row, 2))  # Add the queenside castling move (king's destination)

        return valid_moves

    def is_valid_move(self, selected_piece: str, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> bool:
        """
        Checks if a move from start_pos to end_pos is valid for the selected piece.
        This method is largely redundant if highlight_moves is used correctly,
        but kept for clarity or specific validation needs.
        """
        if selected_piece is None:
            return False

        row, col = start_pos
        valid_moves = self.highlight_moves((row, col))
        return end_pos in valid_moves

    def handle_pawn_promotion(self, row: int, col: int) -> bool:
        """
        Checks if a pawn has reached the opposite end of the board,
        triggering a promotion.
        """
        piece = self.get_piece(row, col)
        if piece and piece[1] == 'P':
            # White pawn promotes on row 0, Black pawn promotes on row 7
            if (piece[0] == 'w' and row == 0) or (piece[0] == 'b' and row == 7):
                return True
        return False

    def promote_pawn(self, row: int, col: int, new_piece_type: str) -> bool:
        """
        Promotes a pawn at (row, col) to the specified new piece type (e.g., 'Q', 'R', 'B', 'N').
        """
        piece = self.get_piece(row, col)
        if piece and piece[1] == 'P':
            color = piece[0]
            self.place_piece(row, col, color + new_piece_type)
            return True
        return False

    def evaluate_board(self) -> int:
        """
        Evaluates the current board position from the perspective of the current player.
        Positive values favor the current player.
        This evaluation function combines material advantage with basic positional considerations.
        """
        score = 0
        current_player_color = self.current_turn

        # Iterate through all squares to sum up piece values and positional scores
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                if piece:
                    piece_type = piece[1]
                    piece_color = piece[0]
                    value = self.piece_values.get(piece_type, 0)

                    # Add positional score based on piece type and its position
                    positional_value = 0
                    # Adjust row for black pieces to mirror the table for white
                    # For white, row r is used directly. For black, 7-r is used.
                    # This ensures the tables are applied correctly from each player's perspective.
                    table_row = r if piece_color == 'w' else 7 - r

                    if piece_type == 'P':
                        positional_value = self.pawn_table[table_row][c]
                    elif piece_type == 'N':
                        positional_value = self.knight_table[table_row][c]
                    elif piece_type == 'B':
                        positional_value = self.bishop_table[table_row][c]
                    elif piece_type == 'R':
                        positional_value = self.rook_table[table_row][c]
                    elif piece_type == 'Q':
                        positional_value = self.queen_table[table_row][c]
                    elif piece_type == 'K':
                        # Use different king tables for midgame and endgame
                        total_pieces = sum(1 for row_pieces in self.board for p in row_pieces if p is not None)
                        if total_pieces < 10: # Arbitrary threshold for endgame
                            positional_value = self.king_table_endgame[table_row][c]
                        else:
                            positional_value = self.king_table_midgame[table_row][c]

                    # Adjust score based on color (positive for current player, negative for opponent)
                    if piece_color == current_player_color:
                        score += value + positional_value
                    else:
                        score -= (value + positional_value)

        # Add bonus for having more captured opponent pieces (material advantage)
        # This part needs careful adjustment for negamax.
        # The evaluation should always be from the perspective of the 'current_turn' player.
        # So, if current_turn is 'w', we want to maximize white's score.
        # If current_turn is 'b', we want to maximize black's score.
        if current_player_color == 'w':
            score += len(self.captured_pieces['w']) * 50 # White captured black pieces
            score -= len(self.captured_pieces['b']) * 50 # Black captured white pieces
        else: # current_player_color == 'b'
            score += len(self.captured_pieces['b']) * 50 # Black captured white pieces
            score -= len(self.captured_pieces['w']) * 50 # White captured white pieces


        # Check for checkmate/stalemate (terminal states)
        # These scores are from the perspective of the player whose turn it is.
        if self.is_checkmate(current_player_color):
            return -99999999 # Current player is checkmated, very bad score
        if self.is_stalemate(current_player_color):
            return 0 # Draw
        
        # If the opponent is checkmated, it's a win for the current player
        opponent_color = 'w' if current_player_color == 'b' else 'b'
        if self.is_checkmate(opponent_color):
            return 99999999 # Opponent is checkmated, very good score

        return score

    def negamax(self, depth: int, alpha: float, beta: float) -> float:
        """
        Negamax algorithm with Alpha-Beta Pruning to find the best move.
        The evaluation is always from the perspective of the current player.

        Args:
            depth (int): The current depth of the search.
            alpha (float): Alpha value for Alpha-Beta Pruning.
            beta (float): Beta value for Beta-Beta Pruning.

        Returns:
            float: The evaluated score of the current board state.
        """
        # Base case: if depth is 0 or game is over, return the board evaluation
        if depth == 0 or self.game_over:
            return self.evaluate_board()

        # Check for terminal states at current depth (important for pruning)
        # These checks are from the perspective of the player whose turn it is.
        if self.is_checkmate(self.current_turn):
            return -float('inf') # Current player is checkmated
        if self.is_stalemate(self.current_turn):
            return 0 # Draw

        max_eval = -float('inf')

        # Store original game state to revert after simulation
        original_board = [row[:] for row in self.board]
        original_kings_moved = self.kings_moved.copy()
        original_rooks_moved = {c: r.copy() for c, r in self.rooks_moved.items()}
        original_en_passant_target = self.en_passant_target
        original_captured_pieces = {c: p[:] for c, p in self.captured_pieces.items()} # Deep copy captured pieces
        original_current_turn = self.current_turn

        # Generate all legal moves for the current player
        all_legal_moves = []
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                if piece and piece[0] == self.current_turn:
                    valid_moves = self.highlight_moves((r, c), check_for_check=True)
                    for move in valid_moves:
                        all_legal_moves.append(((r, c), move))

        # Move Ordering: Prioritize captures
        # A simple heuristic: moves that result in a capture are generally better.
        # Assign a score to each move: higher for captures.
        def move_score(move):
            start_pos, end_pos = move
            captured_piece = self.get_piece(end_pos[0], end_pos[1])
            if captured_piece:
                # Return the value of the captured piece for ordering
                return self.piece_values.get(captured_piece[1], 0)
            return 0 # No capture

        # Sort moves in descending order of their score (captures first)
        all_legal_moves.sort(key=move_score, reverse=True)


        for start_pos, end_pos in all_legal_moves:
            # Simulate the move
            moved_piece = self.get_piece(start_pos[0], start_pos[1])
            captured_piece_at_end_pos = self.get_piece(end_pos[0], end_pos[1]) # Piece at destination before move

            is_en_passant = False
            captured_ep_pawn = None
            # Check for en passant capture during simulation
            if moved_piece and moved_piece[1] == 'P' and end_pos == self.en_passant_target:
                captured_pawn_row = start_pos[0]
                captured_pawn_col = end_pos[1]
                captured_ep_pawn = self.get_piece(captured_pawn_row, captured_pawn_col)
                self.place_piece(captured_pawn_row, captured_pawn_col, None) # Remove captured pawn
                is_en_passant = True
                # Add captured pawn to captured_pieces for correct evaluation
                if captured_ep_pawn:
                    self.captured_pieces[moved_piece[0]].append(captured_ep_pawn)
            elif captured_piece_at_end_pos:
                # Add captured piece to captured_pieces for correct evaluation
                self.captured_pieces[moved_piece[0]].append(captured_piece_at_end_pos)


            # Make the move on the board
            self.place_piece(end_pos[0], end_pos[1], moved_piece)
            self.place_piece(start_pos[0], start_pos[1], None)

            # Handle castling during simulation
            if moved_piece[1] == 'K' and abs(start_pos[1] - end_pos[1]) == 2:
                # Kingside castling
                if end_pos[1] > start_pos[1]:
                    rook_start_col = 7
                    rook_end_col = end_pos[1] - 1
                # Queenside castling
                else:
                    rook_start_col = 0
                    rook_end_col = end_pos[1] + 1
                rook_piece = self.get_piece(start_pos[0], rook_start_col)
                self.place_piece(start_pos[0], rook_end_col, rook_piece)
                self.place_piece(start_pos[0], rook_start_col, None)

            # Handle pawn promotion during simulation (always to Queen for simplicity in AI)
            if moved_piece[1] == 'P' and end_pos[0] == (0 if moved_piece[0] == 'w' else 7):
                self.place_piece(end_pos[0], end_pos[1], moved_piece[0] + 'Q')

            # Update game state for castling and en passant target
            self._update_game_state_after_move(start_pos, end_pos, moved_piece, is_simulation=True)

            # Switch turn for recursive call
            self.current_turn = 'b' if self.current_turn == 'w' else 'w'

            # Recursively call negamax for the opponent (negating the result)
            eval = -self.negamax(depth - 1, -beta, -alpha)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval) # Update alpha

            # Undo the move to restore board state for next sibling move
            self._undo_simulated_move(start_pos, end_pos, moved_piece, captured_piece_at_end_pos, is_en_passant, captured_ep_pawn, original_kings_moved, original_rooks_moved, original_en_passant_target)
            # Restore captured pieces list for the next iteration
            self.captured_pieces = {c: p[:] for c, p in original_captured_pieces.items()}
            self.current_turn = original_current_turn # Restore turn

            # Alpha-Beta Pruning
            if beta <= alpha:
                break # Beta cut-off

        return max_eval

    def _update_game_state_after_move(self, start_pos: tuple[int, int], end_pos: tuple[int, int], moved_piece: str, is_simulation: bool = False):
        """
        Helper to update game state variables like kings_moved, rooks_moved,
        and en_passant_target after a move.
        """
        piece_color = moved_piece[0]
        piece_type = moved_piece[1]

        # Update king/rook moved status for castling
        if piece_type == 'K':
            self.kings_moved[piece_color] = True
        elif piece_type == 'R':
            # Check if the rook is on its original square before updating
            if piece_color == 'w':
                if start_pos == (7, 0): self.rooks_moved['w']['queenside'] = True
                if start_pos == (7, 7): self.rooks_moved['w']['kingside'] = True
            elif piece_color == 'b':
                if start_pos == (0, 0): self.rooks_moved['b']['queenside'] = True
                if start_pos == (0, 7): self.rooks_moved['b']['kingside'] = True

        # Update en passant target square
        self.en_passant_target = None # Reset for the next turn
        if piece_type == 'P' and abs(start_pos[0] - end_pos[0]) == 2:
            # If a pawn moved two squares, set en passant target
            ep_row = (start_pos[0] + end_pos[0]) // 2
            self.en_passant_target = (ep_row, start_pos[1])

        # If this is a real move (not simulation), update turn
        # This part is handled by the negamax function itself by switching self.current_turn
        # before the recursive call and reverting it after.
        # So, this `if not is_simulation` block is not needed here for negamax.
        pass


    def _undo_simulated_move(self, start_pos: tuple[int, int], end_pos: tuple[int, int], moved_piece: str, captured_piece_at_end_pos: str | None, is_en_passant: bool, captured_ep_pawn: str | None, original_kings_moved: dict, original_rooks_moved: dict, original_en_passant_target: tuple[int, int] | None):
        """
        Helper to undo a simulated move, restoring the board and game state variables.
        This is crucial for the negamax algorithm.
        """
        # Revert piece movement
        self.place_piece(start_pos[0], start_pos[1], moved_piece)
        self.place_piece(end_pos[0], end_pos[1], captured_piece_at_end_pos)

        # Revert en passant capture if it happened
        if is_en_passant and captured_ep_pawn:
            captured_pawn_row = start_pos[0] # Pawn was on the same row as the capturing pawn
            captured_pawn_col = end_pos[1] # Pawn was on the same column as the captured square
            self.place_piece(captured_pawn_row, captured_pawn_col, captured_ep_pawn)

        # Revert castling during simulation
        if moved_piece[1] == 'K' and abs(start_pos[1] - end_pos[1]) == 2:
            # Determine if kingside or queenside castling
            if end_pos[1] > start_pos[1]: # Kingside
                rook_start_col = 7
                rook_end_col = end_pos[1] - 1
            else: # Queenside
                rook_start_col = 0
                rook_end_col = end_pos[1] + 1
            # Move rook back to its original position
            rook_piece = self.get_piece(start_pos[0], rook_end_col) # Get the rook from its moved position
            self.place_piece(start_pos[0], rook_start_col, rook_piece)
            self.place_piece(start_pos[0], rook_end_col, None) # Clear the square it moved to

        # Revert pawn promotion during simulation
        if moved_piece[1] == 'P' and end_pos[0] == (0 if moved_piece[0] == 'w' else 7):
            self.place_piece(end_pos[0], end_pos[1], moved_piece[0] + 'P') # Revert to pawn

        # Restore original game state variables
        self.kings_moved = original_kings_moved.copy()
        self.rooks_moved = {c: r.copy() for c, r in original_rooks_moved.items()}
        self.en_passant_target = original_en_passant_target


    def make_ai_move(self) -> bool:
        """
        Determines and executes the best move for the AI (Black) using the Negamax algorithm.
        The search depth is determined by the `ai_difficulty` setting.
        """
        # Determine search depth based on AI difficulty
        depth = 1 # Easy
        if self.ai_difficulty == 'hard':
            depth = 2
        elif self.ai_difficulty == 'pro':
            depth = 2 # Pro: More computationally intensive, adjust as needed for performance

        best_move = None
        max_eval = -float('inf')

        # Initialize alpha-beta for the root node
        alpha = -float('inf')
        beta = float('inf')

        # Store original game state before starting the search
        original_board_state = [row[:] for row in self.board]
        original_kings_moved_state = self.kings_moved.copy()
        original_rooks_moved_state = {c: r.copy() for c, r in self.rooks_moved.items()}
        original_en_passant_target_state = self.en_passant_target
        original_captured_pieces_state = {c: p[:] for c, p in self.captured_pieces.items()} # Deep copy captured pieces
        original_current_turn_state = self.current_turn # This will be 'b' for AI's turn

        all_ai_moves = []
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                if piece and piece[0] == original_current_turn_state: # Consider only AI's pieces
                    valid_moves = self.highlight_moves((r, c), check_for_check=True) # Get legal moves
                    for move in valid_moves:
                        all_ai_moves.append(((r, c), move))

        # Move Ordering for the root node: Prioritize captures
        def root_move_score(move):
            start_pos, end_pos = move
            captured_piece = self.get_piece(end_pos[0], end_pos[1])
            if captured_piece:
                return self.piece_values.get(captured_piece[1], 0)
            return 0

        all_ai_moves.sort(key=root_move_score, reverse=True)

        for start_pos, end_pos in all_ai_moves:
            # Simulate the move
            moved_piece = self.get_piece(start_pos[0], start_pos[1])
            captured_piece_at_end_pos = self.get_piece(end_pos[0], end_pos[1]) # Piece at destination before move

            is_en_passant_capture = False
            captured_ep_pawn = None
            # Check for en passant capture during simulation
            if moved_piece and moved_piece[1] == 'P' and end_pos == original_en_passant_target_state:
                captured_pawn_row = start_pos[0]
                captured_pawn_col = end_pos[1]
                captured_ep_pawn = self.get_piece(captured_pawn_row, captured_pawn_col)
                self.place_piece(captured_pawn_row, captured_pawn_col, None) # Remove captured pawn
                is_en_passant_capture = True
                # Add captured pawn to captured_pieces for correct evaluation
                if captured_ep_pawn:
                    self.captured_pieces[moved_piece[0]].append(captured_ep_pawn)
            elif captured_piece_at_end_pos:
                # Add captured piece to captured_pieces for correct evaluation
                self.captured_pieces[moved_piece[0]].append(captured_piece_at_end_pos)

            self.place_piece(end_pos[0], end_pos[1], moved_piece)
            self.place_piece(start_pos[0], start_pos[1], None)

            # Handle castling during simulation
            if moved_piece[1] == 'K' and abs(start_pos[1] - end_pos[1]) == 2:
                # Kingside castling
                if end_pos[1] > start_pos[1]:
                    rook_start_col = 7
                    rook_end_col = end_pos[1] - 1
                # Queenside castling
                else:
                    rook_start_col = 0
                    rook_end_col = end_pos[1] + 1
                rook_piece = self.get_piece(start_pos[0], rook_start_col)
                self.place_piece(start_pos[0], rook_end_col, rook_piece)
                self.place_piece(start_pos[0], rook_start_col, None)

            # Handle pawn promotion during simulation (always to Queen for simplicity in AI)
            if moved_piece[1] == 'P' and end_pos[0] == (0 if moved_piece[0] == 'w' else 7):
                self.place_piece(end_pos[0], end_pos[1], moved_piece[0] + 'Q')

            # Update game state for castling and en passant target
            self._update_game_state_after_move(start_pos, end_pos, moved_piece, is_simulation=True)

            # Switch turn for recursive call
            self.current_turn = 'b' if self.current_turn == 'w' else 'w'

            # Evaluate the board after this move (recursive call to negamax)
            # The eval is negated because it's from the opponent's perspective
            eval = -self.negamax(depth - 1, -beta, -alpha)

            # If this move leads to a better evaluation for AI, update best_move
            if eval > max_eval:
                max_eval = eval
                best_move = (start_pos, end_pos)

            # Update alpha for alpha-beta pruning at the root
            alpha = max(alpha, eval)

            # Undo the simulated move to restore the board for the next iteration
            self._undo_simulated_move(start_pos, end_pos, moved_piece, captured_piece_at_end_pos, is_en_passant_capture, captured_ep_pawn, original_kings_moved_state, original_rooks_moved_state, original_en_passant_target_state)
            # Restore captured piece list for the next iteration of the root search
            self.captured_pieces = {c: p[:] for c, p in original_captured_pieces_state.items()}
            self.current_turn = original_current_turn_state # Restore turn for next root iteration

            # Alpha-Beta Pruning at the root level
            if beta <= alpha:
                break # Prune remaining moves at this level

        # After finding the best move, execute it on the actual board
        if best_move:
            start_pos, end_pos = best_move
            piece = self.get_piece(start_pos[0], start_pos[1])
            if not piece: # Should not happen if best_move is valid
                return False

            # Store move for undo (full details for the actual game history)
            move_info = {
                'start_pos': start_pos,
                'end_pos': end_pos,
                'moved_piece': piece,
                'captured_piece': self.get_piece(end_pos[0], end_pos[1]), # Piece at end_pos before move
                'promoted_to': None,
                'castling_info': None,
                'en_passant_target_before_move': self.en_passant_target # Store for undo
            }

            # Handle capture (including en passant)
            target_piece = self.get_piece(end_pos[0], end_pos[1])
            if target_piece:
                self.captured_pieces['b'].append(target_piece)

            # Handle en passant capture for the actual move
            is_en_passant_actual = False
            if piece[1] == 'P' and end_pos == self.en_passant_target:
                captured_pawn_row = start_pos[0]
                captured_pawn_col = end_pos[1]
                captured_piece_ep = self.get_piece(captured_pawn_row, captured_pawn_col)
                if captured_piece_ep:
                    self.captured_pieces['b'].append(captured_piece_ep)
                    self.place_piece(captured_pawn_row, captured_pawn_col, None) # Remove captured pawn
                    is_en_passant_actual = True

            move_info['is_en_passant_capture'] = is_en_passant_actual

            # Handle king and rook movement for castling
            if piece[1] == 'K':
                self.kings_moved[piece[0]] = True
                if abs(start_pos[1] - end_pos[1]) == 2: # Castling
                    if end_pos[1] > start_pos[1]: # Kingside castling
                        rook = self.get_piece(start_pos[0], 7)
                        self.place_piece(start_pos[0], end_pos[1] - 1, rook) # Move rook
                        self.place_piece(start_pos[0], 7, None) # Clear old rook square
                        move_info['castling_info'] = {'rook_start': (start_pos[0], 7), 'rook_end': (start_pos[0], end_pos[1] - 1), 'rook_piece': rook}
                    else: # Queenside castling
                        rook = self.get_piece(start_pos[0], 0)
                        self.place_piece(start_pos[0], end_pos[1] + 1, rook) # Move rook
                        self.place_piece(start_pos[0], 0, None) # Clear old rook square
                        move_info['castling_info'] = {'rook_start': (start_pos[0], 0), 'rook_end': (start_pos[0], end_pos[1] + 1), 'rook_piece': rook}
            elif piece[1] == 'R':
                color = piece[0]
                if start_pos[0] == (7 if color == 'w' else 0):
                    if start_pos[1] == 0:  # Queenside rook
                        self.rooks_moved[color]['queenside'] = True
                    elif start_pos[1] == 7:  # Kingside rook
                        self.rooks_moved[color]['kingside'] = True

            # Make the actual piece move on the board
            self.place_piece(end_pos[0], end_pos[1], piece)
            self.place_piece(start_pos[0], start_pos[1], None)

            # Check for pawn promotion (AI always promotes to Queen)
            if piece[1] == 'P' and end_pos[0] == (7 if piece[0] == 'b' else 0): # Black pawn promotes at row 7, White at row 0
                self.promote_pawn(end_pos[0], end_pos[1], 'Q')  # Always promote to queen for AI
                move_info['promoted_to'] = 'Q'

            # Update en passant target for the next turn
            self.en_passant_target = None
            if piece[1] == 'P' and abs(start_pos[0] - end_pos[0]) == 2:
                ep_row = (start_pos[0] + end_pos[0]) // 2
                self.en_passant_target = (ep_row, start_pos[1])

            self.move_history.append(move_info)
            self.redo_history.clear() # Clear redo history when a new move is made
            return True
        return False

    def undo_last_move(self) -> bool:
        """
        Undoes the last move made by either player.
        Restores board state, captured pieces, and special move flags.
        """
        if not self.move_history:
            return False # No moves to undo

        last_move = self.move_history.pop()
        start_pos = last_move['start_pos']
        end_pos = last_move['end_pos']
        moved_piece = last_move['moved_piece']
        captured_piece_at_end_pos = last_move['captured_piece'] # Piece that was at end_pos before move
        promoted_to = last_move['promoted_to']
        castling_info = last_move['castling_info']
        is_en_passant_capture = last_move.get('is_en_passant_capture', False)
        original_en_passant_target_before_move = last_move.get('en_passant_target_before_move')

        # Revert piece movement
        self.place_piece(start_pos[0], start_pos[1], moved_piece)
        self.place_piece(end_pos[0], end_pos[1], captured_piece_at_end_pos)

        # Revert captured piece list
        if captured_piece_at_end_pos:
            capturing_color = moved_piece[0]
            # Only remove if it was an actual capture and the last one added by this color
            # For en passant, the captured piece is added to the capturing player's list.
            # For regular capture, it's also added to the capturing player's list.
            # So, we need to remove it from the capturing player's list.
            if self.captured_pieces[capturing_color] and captured_piece_at_end_pos in self.captured_pieces[capturing_color]:
                self.captured_pieces[capturing_color].remove(captured_piece_at_end_pos)


        # Revert en passant capture
        if is_en_passant_capture:
            # The captured pawn was not at end_pos, but one square behind it
            captured_pawn_row = start_pos[0]
            captured_pawn_col = end_pos[1]
            # The captured piece was the last one added to the capturing player's list
            # We already handled removing it from captured_pieces above.
            # Now, place it back on the board.
            # The `captured_piece_at_end_pos` for en passant is the actual pawn that was captured.
            self.place_piece(captured_pawn_row, captured_pawn_col, captured_piece_at_end_pos)


        # Revert pawn promotion
        if promoted_to:
            # If a pawn was promoted, revert it back to a pawn at its original square
            self.place_piece(end_pos[0], end_pos[1], moved_piece[0] + 'P')
            # The piece at start_pos was the original pawn, ensure it's there
            # But it ensures the original pawn is correctly placed back.
            self.place_piece(start_pos[0], start_pos[1], moved_piece)


        # Revert castling
        if castling_info:
            rook_start = castling_info['rook_start']
            rook_end = castling_info['rook_end']
            rook_piece = castling_info['rook_piece']
            self.place_piece(rook_start[0], rook_start[1], rook_piece) # Move rook back
            self.place_piece(rook_end[0], rook_end[1], None) # Clear rook's moved square

            # Reset kings_moved and rooks_moved for castling
            self.kings_moved[moved_piece[0]] = False
            # Determine if it was kingside or queenside castling to reset rook_moved flag
            if rook_start[1] == 7: # Kingside rook
                self.rooks_moved[moved_piece[0]]['kingside'] = False
            elif rook_start[1] == 0: # Queenside rook
                self.rooks_moved[moved_piece[0]]['queenside'] = False
        else:
            # If not castling, check if king/rook moved and revert their moved status
            # This is a simplification. A more robust solution would check if the king/rook
            # moved *again* later in history. For now, we just revert the flag if this was
            # the move that set it.
            if moved_piece[1] == 'K':
                # Only revert if the king was on its original square before this move
                # and this move was the one that set kings_moved to True.
                # This is hard to track with just the last move.
                # A simpler approach for undo is to just reset the flag.
                self.kings_moved[moved_piece[0]] = False
            elif moved_piece[1] == 'R':
                color = moved_piece[0]
                if start_pos[0] == (7 if color == 'w' else 0):
                    if start_pos[1] == 0:  # Queenside rook
                        self.rooks_moved[color]['queenside'] = True
                    elif start_pos[1] == 7:  # Kingside rook
                        self.rooks_moved[color]['kingside'] = True

        # Restore the en passant target from before the move
        self.en_passant_target = original_en_passant_target_before_move

        # Change turn back
        self.current_turn = 'b' if self.current_turn == 'w' else 'w'
        self.game_over = False # If game was over, it's not anymore
        self.winner = None
        self.redo_history.append(last_move) # Store the undone move for redo
        return True

    def redo_last_move(self) -> bool:
        """
        Redoes the last undone move.
        Restores board state, captured pieces, and special move flags.
        """
        if not self.redo_history:
            return False # No moves to redo

        redone_move = self.redo_history.pop()
        start_pos = redone_move['start_pos']
        end_pos = redone_move['end_pos']
        moved_piece = redone_move['moved_piece']
        captured_piece_at_end_pos = redone_move['captured_piece']
        promoted_to = redone_move['promoted_to']
        castling_info = redone_move['castling_info']
        is_en_passant_capture = redone_move.get('is_en_passant_capture', False)
        original_en_passant_target_before_move = redone_move.get('en_passant_target_before_move')

        # Re-apply piece movement
        self.place_piece(end_pos[0], end_pos[1], moved_piece)
        self.place_piece(start_pos[0], start_pos[1], None)

        # Re-add captured piece to list
        if captured_piece_at_end_pos:
            capturing_color = moved_piece[0]
            self.captured_pieces[capturing_color].append(captured_piece_at_end_pos)

        # Re-apply en passant capture
        if is_en_passant_capture:
            captured_pawn_row = start_pos[0]
            captured_pawn_col = end_pos[1]
            self.place_piece(captured_pawn_row, captured_pawn_col, None) # Remove the captured pawn again

        # Re-apply pawn promotion
        if promoted_to:
            self.place_piece(end_pos[0], end_pos[1], moved_piece[0] + promoted_to)

        # Re-apply castling
        if castling_info:
            rook_start = castling_info['rook_start']
            rook_end = castling_info['rook_end']
            rook_piece = castling_info['rook_piece']
            self.place_piece(rook_end[0], rook_end[1], rook_piece) # Move rook back to its new position
            self.place_piece(rook_start[0], rook_start[1], None) # Clear old rook square

            # Re-set kings_moved and rooks_moved for castling
            self.kings_moved[moved_piece[0]] = True
            if rook_start[1] == 7: # Kingside rook
                self.rooks_moved[moved_piece[0]]['kingside'] = True
            elif rook_start[1] == 0: # Queenside rook
                self.rooks_moved[moved_piece[0]]['queenside'] = True
        else:
            # Re-set king/rook moved flags if this move was their first
            if moved_piece[1] == 'K':
                self.kings_moved[moved_piece[0]] = True
            elif moved_piece[1] == 'R':
                color = moved_piece[0]
                if start_pos[0] == (7 if color == 'w' else 0):
                    if start_pos[1] == 0:  # Queenside rook
                        self.rooks_moved[color]['queenside'] = True
                    elif start_pos[1] == 7:  # Kingside rook
                        self.rooks_moved[color]['kingside'] = True

        # Restore the en passant target for the next turn (this is tricky, needs to be the target *after* this move)
        # For redo, the en_passant_target should be set based on the move that was just redone.
        # If the redone move was a double pawn push, set the target. Otherwise, clear it.
        self.en_passant_target = None
        if moved_piece[1] == 'P' and abs(start_pos[0] - end_pos[0]) == 2:
            ep_row = (start_pos[0] + end_pos[0]) // 2
            self.en_passant_target = (ep_row, start_pos[1])

        self.move_history.append(redone_move) # Add back to move history
        self.current_turn = 'b' if self.current_turn == 'w' else 'w' # Change turn back
        self.game_over = False
        self.winner = None
        return True


# --- Pawn Promotion Dialog Class ---
class PawnPromotionDialog(QDialog):
    """
    A dialog window that appears when a pawn reaches the opposite end of the board,
    allowing the player to choose which piece to promote the pawn to.
    """
    def __init__(self, color: str, parent=None):
        """
        Initializes the pawn promotion dialog.

        Args:
            color (str): The color of the pawn ('w' or 'b').
            parent (QWidget): The parent widget.
        """
        super().__init__(parent)
        self.color = color
        self.selected_piece = 'Q'  # Default to Queen if dialog is closed without selection
        self.setWindowTitle("Pawn Promotion")
        self.setModal(True) # Make it a modal dialog (blocks parent window)

        # Use the local logo for the dialog icon
        logo_path = download_image_if_not_exists(
            "logo.png",
            "logo"
        )
        if logo_path:
            self.setWindowIcon(QIcon(logo_path))
        else:
            self.setWindowIcon(QIcon()) # Set empty icon if not found

        layout = QGridLayout()
        layout.setSpacing(10) # Add spacing between buttons

        # Pieces available for promotion
        pieces = ['Q', 'R', 'B', 'N']
        piece_names = ['Queen', 'Rook', 'Bishop', 'Knight']

        # Add piece buttons to the dialog
        for i, (piece_type_code, name) in enumerate(zip(pieces, piece_names)):
            button = QPushButton(name)
            button.setFixedSize(150, 60) # Slightly wider to accommodate text and icon
            button.setFont(QFont("Cinzel", 10, QFont.Bold)) # Slightly smaller font for better fit

            piece_code_full = color + piece_type_code # e.g., 'wQ', 'bR'

            current_piece_set_dir = "images1" # Default fallback
            if parent and hasattr(parent, 'current_piece_set_dir'):
                current_piece_set_dir = parent.current_piece_set_dir

            img_path = download_image_if_not_exists(
                f"{piece_code_full}.png",
                current_piece_set_dir # Use the current piece set directory
            )
            if img_path:
                button.setIcon(QIcon(img_path))
                button.setIconSize(QSize(40, 32)) # Smaller icon size
                button.setText(name) # Ensure text is explicitly set
            else:
                # If image not found, ensure only text is displayed
                button.setText(name)
                button.setIcon(QIcon()) # Clear any default icon

            # Connect button click to select_piece method
            button.clicked.connect(lambda _, p=piece_type_code: self.select_piece(p))
            layout.addWidget(button, 0, i) # All buttons in the first row

        self.setLayout(layout)
        self.setFixedSize(layout.sizeHint()) # Adjust dialog size to fit content

    def select_piece(self, piece: str):
        """
        Sets the selected piece type for promotion and accepts the dialog.
        """
        self.selected_piece = piece
        self.accept()

# New Settings Dialog Class
class SettingsDialog(QDialog):
    def __init__(self, parent=None, initial_volume: int = 50):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(300, 150) # Adjust size as needed

        logo_path = download_image_if_not_exists("logo.png", "logo")
        if logo_path:
            self.setWindowIcon(QIcon(logo_path))

        layout = QVBoxLayout()

        # Volume Controls
        volume_label = QLabel("Volume:")
        volume_label.setObjectName("volumeLabel") # Use existing QSS style
        layout.addWidget(volume_label)

        volume_layout = QHBoxLayout()
        self.volume_down_button = QPushButton("-")
        self.volume_down_button.setObjectName("volumeDownButton") # Use existing QSS style
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setObjectName("volumeSlider") # Use existing QSS style
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(initial_volume)
        self.volume_up_button = QPushButton("+")
        self.volume_up_button.setObjectName("volumeUpButton") # Use existing QSS style

        volume_layout.addWidget(self.volume_down_button)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_up_button)
        layout.addLayout(volume_layout)

        # Connect signals
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_up_button.clicked.connect(self.increase_volume)
        self.volume_down_button.clicked.connect(self.decrease_volume)

        self.setLayout(layout)

    def set_volume(self, value: int):
        # Emit a signal that the main window can connect to
        if self.parent() and hasattr(self.parent(), 'set_global_volume'):
            self.parent().set_global_volume(value)

    def increase_volume(self):
        current_volume = self.volume_slider.value()
        new_volume = min(100, current_volume + 10)
        self.volume_slider.setValue(new_volume)

    def decrease_volume(self):
        current_volume = self.volume_slider.value()
        new_volume = max(0, current_volume - 10)
        self.volume_slider.setValue(new_volume)


# --- Main Chess Board GUI Class ---
class ChessBoard(QMainWindow):
    """
    The main GUI class for the chess game, handling the display, user interaction,
    and integration with the ChessLogic.
    """
    def __init__(self, chess_logic: ChessLogic):
        """
        Initializes the ChessBoard GUI.

        Args:
            chess_logic (ChessLogic): An instance of the ChessLogic class to manage game state.
        """
        super().__init__()
        self.chess_logic = chess_logic
        self.valid_moves = []  # Stores valid moves for the currently selected piece
        self.game_mode = "two_player"  # Default game mode
        self.ai_thinking = False # Flag to prevent user input during AI turn
        self.use_unicode_pieces = False # Flag to use unicode symbols if image loading fails

        # Store current board colors for refresh_board
        self.current_light_square_color = QColor(255, 255, 255)
        self.current_dark_square_color = QColor(169, 169, 169)

        # Store current piece set directory
        self.current_piece_set_dir = "images1" # Default piece set directory

        # --- Window Setup ---
        self.setWindowTitle("𝚺 CHESS")
        self.setGeometry(50, 30, 1000, 800) # x, y, width, height

        # Load and set the application icon (logo)
        logo_path = download_image_if_not_exists(
            "logo.png",
            "logo"
        )
        if logo_path:
            self.setWindowIcon(QIcon(logo_path))
            self.logo_pixmap = QPixmap(logo_path)
        else:
            self.setWindowIcon(QIcon()) # Set empty icon if not found
            self.logo_pixmap = None # No logo pixmap available

        # Apply custom QSS styling
        self.setStyleSheet(self.get_embedded_qss())

        # --- Main Layout Structure ---
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_vertical_layout = QVBoxLayout(self.centralwidget)

        # Initialize the QWebEngineView for the homepage
        self.homepage_view = None
        self.welcome_popup_view = None # Initialize popup view

        # Initialize QMediaPlayer for sound effects
        self.media_player = QMediaPlayer(self)
        self.move_sound_url = get_local_file_url("move_sound.mp3", "sounds")
        self.capture_sound_url = get_local_file_url("capture_sound.mp3", "sounds") # New: Capture sound
        self.check_sound_url = get_local_file_url("check_sound.mp3", "sounds") # New: Check sound

        # Check if sound files exist
        if not self.move_sound_url.isValid() or not os.path.exists(self.move_sound_url.toLocalFile()):
            print(f"Warning: Move sound file not found at {self.move_sound_url.toLocalFile()}. Move sound effects will be disabled.")
            self.move_sound_url = None
        if not self.capture_sound_url.isValid() or not os.path.exists(self.capture_sound_url.toLocalFile()):
            print(f"Warning: Capture sound file not found at {self.capture_sound_url.toLocalFile()}. Capture sound effects will be disabled.")
            self.capture_sound_url = None
        if not self.check_sound_url.isValid() or not os.path.exists(self.check_sound_url.toLocalFile()):
            print(f"Warning: Check sound file not found at {self.check_sound_url.toLocalFile()}. Check sound effects will be disabled.")
            self.check_sound_url = None

        # Store global volume setting
        self.global_volume = 50 # Default volume

        if WEBENGINE_AVAILABLE:
            from PyQt5.QtWebEngineWidgets import QWebEngineView  # Ensure QWebEngineView is defined
            self.homepage_view = QWebEngineView(self)
            self.main_vertical_layout.addWidget(self.homepage_view)
            self.load_homepage()
        else:
            # If WebEngine is not available, directly show the game UI
            # and inform the user.
            QMessageBox.warning(self, "Warning", "PyQtWebEngine is not installed. "
                                                 "The interactive homepage and custom HTML popup will not be displayed. "
                                                 "Starting directly in Two Player mode.")
            # Default to two-player if homepage is not available
            self.game_mode = "two_player"
            # Proceed directly to show game UI
            self.game_ui_widget = QWidget()
            self.game_ui_layout = QVBoxLayout(self.game_ui_widget)
            self.main_vertical_layout.addWidget(self.game_ui_widget)
            self.setup_game_ui()
            self.game_ui_widget.show()
            # If no webengine, we can use a simple QMessageBox as a fallback for the greeting
            self.show_fallback_greeting()
            self.status_label.setText("Two Player Mode - White's turn") # Initial status

        if self.homepage_view: # This block runs if WEBENGINE_AVAILABLE is True
            # Game UI elements (initially hidden)
            self.game_ui_widget = QWidget()
            self.game_ui_layout = QVBoxLayout(self.game_ui_widget)
            self.main_vertical_layout.addWidget(self.game_ui_widget)
            self.setup_game_ui() # Setup UI elements regardless of initial visibility
            self.game_ui_widget.hide() # Hide game UI initially
            self.show_startup_greeting() # Show the custom HTML greeting popup

        # Board dimensions (fixed for 8x8 chess)
        self.square_size = 80.4 # Size of each square in pixels
        self.board_width = 8
        self.board_height = 8

        # Set up the chess piece images (loads from local or downloads)
        self.setup_piece_images(self.current_piece_set_dir) # Initial load with default directory

        # Initial board setup and display
        self.create_standard_chessboard() # This will set initial colors and call place_pieces
        self.update_captured_pieces_display()

        # --- AI Timer ---
        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self.trigger_ai_move) # Renamed for clarity

    @pyqtSlot(str) # Decorator to expose this method to QWebChannel
    def set_game_mode(self, mode: str):
        """
        Sets the game mode (Two Player or vs AI) and updates the UI accordingly.
        This method is called from JavaScript in the homepage.

        Args:
            mode (str): 'two_player' or 'vs_ai'.
        """
        self.game_mode = mode
        self.show_game_ui() # Switch to game UI

        if mode == "two_player":
            self.status_label.setText("Two Player Mode - White's turn")
            self.undo_button.hide() # Hide undo button for two-player
            self.redo_button.hide() # Hide redo button for two-player
            # Hide AI difficulty options
            self.ai_difficulty_label.hide()
            self.easy_ai_button.hide()
            self.hard_ai_button.hide()
            self.pro_ai_button.hide()
        else: # 'vs_ai' mode
            # Update status label to reflect current AI difficulty
            self.status_label.setText(f"Playing against AI ({self.chess_logic.ai_difficulty.capitalize()}) - White's turn")
            self.undo_button.show() # Show undo button for AI mode
            self.redo_button.show() # Show redo button for AI mode
            # Show AI difficulty options
            self.ai_difficulty_label.show()
            self.easy_ai_button.show()
            self.hard_ai_button.show()
            self.pro_ai_button.show()

        # Reset game when changing mode to apply new settings
        self.reset_game()

    def show_builder_info(self):
        """Displays a QMessageBox with information about the builder."""
        info_box = QMessageBox(self)
        info_box.setWindowTitle("About the Builder")
        info_box.setText("<h3>ΣCHESS</h3>"
                         "<p>This chess application was built by <strong>Arka Das</strong>.</p>"
                         "<p>It features a customizable board, AI opponent, and classic chess gameplay.</p>"
                         "<p>Enjoy your game!</p>")
        info_box.setIcon(QMessageBox.Information)
        info_box.setStandardButtons(QMessageBox.Ok)
        info_box.exec_()

    def show_settings_dialog(self):
        """Displays the settings dialog with volume controls."""
        settings_dialog = SettingsDialog(self, initial_volume=self.global_volume)
        settings_dialog.exec_()

    @pyqtSlot(int)
    def set_global_volume(self, value: int):
        """Sets the global volume for the media player."""
        self.global_volume = value
        self.media_player.setVolume(self.global_volume)


    def load_homepage(self):
        """Loads the HTML content for the homepage into the QWebEngineView."""
        if not WEBENGINE_AVAILABLE or not self.homepage_view:
            print("WebEngine not available, cannot load homepage.")
            return

        logo_path = download_image_if_not_exists("logo.png", "logo")
        if logo_path:
            # Convert local path to a URL for QWebEngineView
            logo_url = QUrl.fromLocalFile(logo_path).toString()
        else:
            logo_url = "https://placehold.co/100x100/e9ecef/495057?text=Logo" # Fallback placeholder

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>𝚺CHESS - Home</title>
            <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Cinzel', Inter;
                    background: linear-gradient(135deg, #f0f4f8, #cdd4da); /* Subtle gradient background */
                    color: #2c3e50; /* Dark grey text */
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    overflow: hidden; /* Prevent scrollbars */
                    position: relative; /* Needed for absolute positioning of pin button */
                }}

                .card {{
                    background-color: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                    padding: 60px;
                    text-align: center;
                    max-width: 550px;
                    width: 96%;
                    transform: translateY(0);
                    transition: transform 0.5s ease-out, box-shadow 2.3s ease-in-out;
                    animation: fadeInScale 0.8s ease-out forwards;
                }}

                .card:hover {{
                    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
                }}

                @keyframes fadeInScale {{
                    from {{ opacity: 0; transform: scale(0.9) translateY(20px); }}
                    to {{ opacity: 1; transform: scale(1) translateY(0); }}
                }}

                .logo-container {{
                    margin-bottom: 25px;
                }}

                .logo {{
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    object-fit: cover;
                    border: 4px solid #3498db;
                    box-shadow: 0 0 15px rgba(52, 152, 219, 0.5);
                }}

                h1 {{
                    font-size: 2.8em;
                    color: #34495e;
                    margin-bottom: 10px;
                    font-weight: 700;
                    letter-spacing: -1px;
                }}

                .tagline {{
                    font-size: 1.1em;
                    color: #7f8c8d;
                    margin-bottom: 35px;
                    line-height: 1.6;
                }}

                .button-group {{
                    display: flex;
                    flex-direction: column;
                    gap: 15px; /* Space between buttons */
                }}

                .action-button {{
                    background: linear-gradient(45deg, #3498db, #2980b9); /* Blue gradient */
                    color: white;
                    padding: 16px 25px;
                    border: none;
                    border-radius: 50px; /* Pill-shaped buttons */
                    font-size: 1.15em;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
                }}

                .action-button:hover {{
                    background: linear-gradient(45deg, #2980b9, #3498db);
                    transform: translateY(-3px);
                    box-shadow: 0 8px 20px rgba(52, 152, 219, 0.4);
                }}

                .action-button:active {{
                    transform: translateY(0);
                    box-shadow: 0 3px 10px rgba(52, 152, 219, 0.3);
                }}

                .action-button.secondary {{
                    background: linear-gradient(45deg, #2ecc71, #27ae60); /* Green gradient */
                    box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
                }}

                .action-button.secondary:hover {{
                    background: linear-gradient(45deg, #27ae60, #2ecc71);
                    box-shadow: 0 8px 20px rgba(46, 204, 113, 0.4);
                }}

                /* Pin button styling */
                .pin-button {{
                    position: absolute; /* Position absolutely within the body */
                    top: 20px; /* 20px from the top */
                    right: 20px; /* 20px from the right */
                    background: linear-gradient(45deg, #f39c12, #e67e22); /* Orange gradient */
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 50px;
                    font-size: 1em;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 5px 15px rgba(243, 156, 18, 0.3);
                    z-index: 1000; /* Ensure it's on top of other elements */
                }}

                .pin-button:hover {{
                    background: linear-gradient(45deg, #e67e22, #f39c12);
                    transform: translateY(-3px);
                    box-shadow: 0 8px 20px rgba(243, 156, 18, 0.4);
                }}

                footer {{
                    margin-top: 40px;
                    color: #95a5a6;
                    font-size: 0.85em;
                }}
            </style>
            <!-- IMPORTANT: Include qwebchannel.js -->
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        </head>
        <body>

            <div class="card">
                <div class="logo-container">
                    <img src="{logo_url}" alt="ChessMaster Logo" class="logo"
                         onerror="this.style.display='none'; document.getElementById('logo-error').style.display='block';">
                    <div id="logo-error" style="display:none; color: #e74c3c; font-size: 0.9em; margin-top: 10px;">
                        Error loading logo.
                    </div>
                </div>

                <h1>𝚺CHESS</h1>
                <p class="tagline">The ultimate chess experience awaits you.</p>

                <div class="button-group">
                    <button class="action-button" id="playHumanBtn">Play with a Human</button>
                    <button class="action-button secondary" id="playAIBtn">Challenge the AI</button>
                </div>
                <footer>
                    &copy; 2025 ΣCHESS. All rights reserved.
                </footer>
            </div>
            <button class="pin-button" id="pinBtn">☰</button> <!-- Three-pin button moved outside card -->

            <script>
                // Wait for the QWebChannel to be ready
                new QWebChannel(qt.webChannelTransport, function(channel) {{
                    // Expose the Python object to JavaScript
                    window.pyqt_obj = channel.objects.pyqt_obj;

                    // Add event listeners to buttons after pyqt_obj is available
                    document.getElementById('playHumanBtn').onclick = function() {{
                        if (window.pyqt_obj) {{
                            window.pyqt_obj.set_game_mode('two_player');
                        }} else {{
                            console.error("pyqt_obj is not defined when 'Play with Human' button clicked.");
                        }}
                    }};

                    document.getElementById('playAIBtn').onclick = function() {{
                        if (window.pyqt_obj) {{
                            window.pyqt_obj.set_game_mode('vs_ai');
                        }} else {{
                            console.error("pyqt_obj is not defined when 'Challenge the AI' button clicked.");
                        }}
                    }};

                    document.getElementById('pinBtn').onclick = function() {{
                        if (window.pyqt_obj) {{
                            window.pyqt_obj.show_pin_menu();
                        }} else {{
                            console.error("pyqt_obj is not defined when 'Pin' button clicked.");
                        }}
                    }};
                }});
            </script>

        </body>
        </html>
        """
        # Use QUrl.fromLocalFile to correctly resolve paths for local files
        # The base URL is important for relative paths in HTML (like the logo)
        base_url = QUrl.fromLocalFile(os.path.abspath(os.path.dirname(__file__)) + os.sep)
        self.homepage_view.setHtml(html_content, base_url)

        self.homepage_view.page().profile().setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
        self.homepage_view.page().profile().setPersistentStoragePath("")
        self.homepage_view.page().profile().setCachePath("")
        self.homepage_view.page().profile().clearHttpCache()

        # Ensure self.channel is initialized before registering objects
        # This part was already correct, but moved after setHtml for logical flow
        self.channel = QWebChannel()
        self.homepage_view.page().setWebChannel(self.channel)
        self.channel.registerObject("pyqt_obj", self)


    @pyqtSlot()
    def show_pin_menu(self):
        """
        Displays a QMenu when the 'Pin' button on the homepage is clicked.
        """
        menu = QMenu(self)
        builder_info_action = QAction("Builder Info", self)
        builder_info_action.triggered.connect(self.show_builder_info)
        menu.addAction(builder_info_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        menu.addAction(settings_action)

        # Get the position of the mouse click to show the menu there
        # This requires getting the mouse position from the QWebEngineView context,
        # which is not directly available via a simple @pyqtSlot.
        # A simpler approach for now is to show it at a fixed position or center.
        # For a more precise context menu, you'd need to pass coordinates from JS.
        # For simplicity, let's show it at the top-right corner of the main window.
        # Or, if we want it near the button, we'd need to map the button's position.
        # Since the button is in the QWebEngineView, we can't directly get its QWidget position.
        # A workaround is to show it at the top-right of the main window.
        # Or, if we want it near the button, we'd need to pass coordinates from JS.
        # For now, let's show it at the top-right of the main window.
        # Get the geometry of the main window
        main_window_rect = self.geometry()
        # Calculate a position near the top-right corner
        # Adjust these values as needed for desired offset from corner
        x = main_window_rect.x() + main_window_rect.width() - menu.sizeHint().width() - 10
        y = main_window_rect.y() + 10
        menu.exec_(QPoint(x, y))


    def show_homepage(self):
        """Shows the homepage and hides the game UI."""
        if WEBENGINE_AVAILABLE and self.homepage_view:
            self.homepage_view.show()
            self.game_ui_widget.hide()
            self.reset_game() # Reset game state when returning to home
        else:
            QMessageBox.information(self, "Information", "Homepage is not available because PyQtWebEngine is not installed.")

    def hide_welcome_popup(self):
        """Hides and deletes the welcome popup."""
        if self.welcome_popup_view:
            # Instead of hiding immediately, trigger the fade-out animation in JS, then hide after a delay
            if hasattr(self, "welcome_popup_view") and self.welcome_popup_view:
                try:
                    page = self.welcome_popup_view.page()
                    if page is not None:
                        page.runJavaScript("triggerFadeOut();")
                except Exception as e:
                    print(f"Error triggering fade-out: {e}")
                QTimer.singleShot(2000, self.welcome_popup_view.hide)  # Hide after 2s for fade-out
            self.welcome_popup_view.deleteLater() # Mark for deletion
            self.welcome_popup_view = None

    def show_startup_greeting(self):
        """
        Displays a custom HTML greeting pop-up after application launch,
        which automatically disappears after 2 seconds.
        """
        if not WEBENGINE_AVAILABLE:
            # Fallback to simple QMessageBox if WebEngine is not available for custom popup
            self.show_fallback_greeting()
            return

        self.welcome_popup_view = QWebEngineView()
        self.welcome_popup_view.setWindowFlags(
            Qt.SplashScreen | # Makes it frameless and floats on top
            Qt.WindowStaysOnTopHint # Ensures it stays on top
        )
        self.welcome_popup_view.setAttribute(Qt.WA_TranslucentBackground) # Make background transparent for rounded corners

        # Adjust size and position of the popup
        popup_width = 400
        popup_height = 250
        screen_geo = QApplication.primaryScreen().geometry()
        center_x = screen_geo.width() // 2 - popup_width // 2
        center_y = screen_geo.height() // 2 - popup_height // 2
        self.welcome_popup_view.setGeometry(center_x, center_y, popup_width, popup_height)


        logo_path = download_image_if_not_exists("logo.png", "logo")
        if logo_path:
            logo_url = QUrl.fromLocalFile(logo_path).toString()
        else:
            logo_url = "https://placehold.co/100x100/e9ecef/495057?text=Logo"

        # HTML content for the pop-up - Simplified JavaScript
        popup_html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome!</title>
            <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background-color: transparent; /* Essential for WA_TranslucentBackground */
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    overflow: hidden;
                }}
                .popup-container {{
                    background: linear-gradient(135deg, #fdfbfb, #ebedee); /* Light gradient */
                    border-radius: 15px;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                    padding: 25px 30px;
                    text-align: center;
                    width: 100%;
                    max-width: 380px; /* Max width for content within the popup_width */
                    box-sizing: border-box;
                    animation: fadeIn 0.5s ease-out forwards;
                    opacity: 0; /* Start hidden for animation */
                    transition: opacity 0.5s ease-in-out; /* For fade out */
                }}
                .popup-container.fade-out {{
                    opacity: 0;
                }}
                .logo-small {{
                    width: 70px;
                    height: 70px;
                    border-radius: 50%;
                    object-fit: cover;
                    border: 3px solid #f39c12; /* GoDaddy-like orange border */
                    box-shadow: 0 0 10px rgba(243, 156, 18, 0.5);
                    margin-bottom: 15px;
                }}
                h2 {{
                    font-family: 'Cinzel', Inter;
                    color: #34495e;
                    margin-top: 0;
                    margin-bottom: 10px;
                    font-size: 1.8em;
                    font-weight: 700;
                }}
                p {{
                    font-family: 'Inter', sans-serif;
                    color: #555;
                    font-size: 1.0em;
                    line-height: 1.5;
                    margin-bottom: 0;
                }}
            </style>
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        </head>
        <body>
            <div class="popup-container" id="popup">
                <img src="{logo_url}" alt="Logo" class="logo-small"
                     onerror="this.src='https://drive.google.com/file/d/1BTP2keqx8xc8__mqKKDCNE8LmnDjKRYR/view?usp=drive_link';">
                <h2>Welcome!
                </h2>
                <p>Prepare for an epic chess battle! </p>
                
                 </p>Build by Arka Das.</p>
            </div>
            <script>
                // This function will be called by Python to trigger fade-out
                function triggerFadeOut() {{
                    const popup = document.getElementById('popup');
                    if (popup) {{
                        popup.classList.add('fade-out');
                    }}
                }}

                // Initial fade-in animation starts
                document.addEventListener('DOMContentLoaded', () => {{
                    document.getElementById('popup').style.opacity = 1;
                }});
            </script>
        </body>
        </html>
        """
        base_url = QUrl.fromLocalFile(os.path.abspath(os.path.dirname(__file__)) + os.sep)
        self.welcome_popup_view.setHtml(popup_html_content, base_url)

        self.welcome_popup_view.show()

        # Start a QTimer to trigger the fade-out animation after 1.5 seconds (2 seconds total display - 0.5s CSS transition)
        self.popup_timer = QTimer(self)
        self.popup_timer.setSingleShot(True)
        self.popup_timer.timeout.connect(lambda: self.welcome_popup_view.page().runJavaScript("triggerFadeOut();"))
        self.popup_timer.start(1000) # Trigger JS fade-out after 1.5 seconds (1500 ms)

        # Start a second QTimer to actually hide and dispose of the QWebEngineView after the full 2 seconds
        QTimer.singleShot(3800, self.hide_welcome_popup) # Close after 2 seconds (2000 ms)


    def show_fallback_greeting(self):
        """
        Displays a simple QMessageBox greeting if QWebEngineView is not available.
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Welcome to Σ CHESS")
        if self.logo_pixmap:
            msg_box.setIconPixmap(self.logo_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Welcome to Σ CHESS!\n\nPrepare for an epic chess battle!")
        msg_box.setStandardButtons(QMessageBox.Ok)
        # Use a QTimer to close the QMessageBox automatically
        QTimer.singleShot(3800, msg_box.accept) # Close after 2 seconds (2000 ms)
        msg_box.exec_()


    def show_game_ui(self):
        """Hides the homepage and shows the game UI."""
        if WEBENGINE_AVAILABLE and self.homepage_view:
            self.homepage_view.hide()
        self.game_ui_widget.show()


    def setup_game_ui(self):
        """
        Sets up the layout and widgets for the main game UI.
        This is separated so it can be called whether the homepage is used or not.
        """
        # Top Banner: Game Mode and Chessboard Styles
        self.top_banner_layout = QHBoxLayout()
        self.game_ui_layout.addLayout(self.top_banner_layout)

        # Board style menu button
        self.board_styles_menu_button = QPushButton("Board Styles", self)
        self.board_styles_menu_button.setObjectName("boardStylesMenuButton")
        self.top_banner_layout.addWidget(self.board_styles_menu_button)

        # Create a QMenu for board styles
        self.board_styles_menu = QMenu(self)
        self.standard_action = self.board_styles_menu.addAction("Standard")
        self.vintage_action = self.board_styles_menu.addAction("Vintage")
        self.dark_action = self.board_styles_menu.addAction("Dark")
        self.high_contrast_action = self.board_styles_menu.addAction("High Contrast")
        self.board_styles_menu.addSeparator() # Separator for custom option
        self.custom_style_action = self.board_styles_menu.addAction("Custom....")

        # Set the menu to the button
        self.board_styles_menu_button.setMenu(self.board_styles_menu)

        # Connect menu actions to methods
        self.standard_action.triggered.connect(self.create_standard_chessboard)
        self.vintage_action.triggered.connect(self.create_styled_chessboard1)
        self.dark_action.triggered.connect(self.create_styled_chessboard2)
        self.high_contrast_action.triggered.connect(self.create_alternative_chessboard)
        self.custom_style_action.triggered.connect(self.show_custom_color_dialog)

        # Piece style menu button
        self.piece_styles_menu_button = QPushButton("Piece Styles", self)
        self.piece_styles_menu_button.setObjectName("pieceStylesMenuButton")
        self.top_banner_layout.addWidget(self.piece_styles_menu_button)

        # Create a QMenu for piece styles
        self.piece_styles_menu = QMenu(self)
        self.piece_set1_action = self.piece_styles_menu.addAction("Classic Set")
        self.piece_set2_action = self.piece_styles_menu.addAction("Modern Set") # Or other descriptive names

        # Set the menu to the button
        self.piece_styles_menu_button.setMenu(self.piece_styles_menu)

        # Connect menu actions to methods
        self.piece_set1_action.triggered.connect(lambda: self.set_piece_set("images1"))
        self.piece_set2_action.triggered.connect(lambda: self.set_piece_set("images2"))

        # Return to Home Button
        self.return_home_button = QPushButton("Return to Home", self)
        self.return_home_button.setObjectName("returnHomeButton")
        self.top_banner_layout.addWidget(self.return_home_button)
        # Only connect if homepage is available
        if WEBENGINE_AVAILABLE:
            self.return_home_button.clicked.connect(self.show_homepage)
        else:
            self.return_home_button.hide() # Hide if no homepage to return to


        # Main horizontal layout for chessboard and right panel
        self.main_horizontal_layout = QHBoxLayout()
        self.game_ui_layout.addLayout(self.main_horizontal_layout)

        # Left side: Chessboard and Status Label
        self.left_panel_layout = QVBoxLayout()
        self.main_horizontal_layout.addLayout(self.left_panel_layout)

        # Game status display
        self.status_label = QLabel("White's turn")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Cinzel", 15, QFont.Bold))
        self.left_panel_layout.addWidget(self.status_label)

        # QGraphicsScene and QGraphicsView for the chessboard rendering
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)

        self.left_panel_layout.addWidget(self.view)
        self.left_panel_layout.addStretch(1) # Push board to top

        # Right side: Controls and Captured Pieces
        self.right_panel_layout = QVBoxLayout()
        self.main_horizontal_layout.addLayout(self.right_panel_layout)
        self.right_panel_layout.setSpacing(15) # Add spacing between sections

        # Captured pieces display
        self.captured_frame = QFrame()
        self.captured_frame.setObjectName("capturedFrame")
        cap_layout = QVBoxLayout(self.captured_frame)
        cap_layout.setContentsMargins(10, 10, 10, 10) # Padding inside frame

        self.white_caps_label = QLabel("White Captured:")
        self.white_caps_label.setObjectName("capturedLabel")
        self.white_caps_display = QHBoxLayout()
        self.white_caps_container = QWidget()
        self.white_caps_container.setLayout(self.white_caps_display)
        self.white_caps_display.setAlignment(Qt.AlignmentFlag.AlignLeft) # Align captured pieces to left

        self.black_caps_label = QLabel("Black Captured:")
        self.black_caps_label.setObjectName("capturedLabel")
        self.black_caps_display = QHBoxLayout()
        self.black_caps_container = QWidget()
        self.black_caps_container.setLayout(self.black_caps_display)
        self.black_caps_display.setAlignment(Qt.AlignmentFlag.AlignLeft) # Align captured pieces to left

        cap_layout.addWidget(self.white_caps_label)
        cap_layout.addWidget(self.white_caps_container)
        cap_layout.addWidget(self.black_caps_label)
        cap_layout.addWidget(self.black_caps_container)
        self.right_panel_layout.addWidget(self.captured_frame)

        # AI Difficulty Selection
        self.ai_difficulty_label = QLabel("AI Difficulty:")
        self.ai_difficulty_label.setObjectName("aiDifficultyLabel")
        self.ai_difficulty_layout = QHBoxLayout()
        self.easy_ai_button = QPushButton("Easy", self)
        self.easy_ai_button.setObjectName("easyAiButton")
        self.hard_ai_button = QPushButton("Hard", self)
        self.hard_ai_button.setObjectName("hardAiButton")
        self.pro_ai_button = QPushButton("Pro", self)
        self.pro_ai_button.setObjectName("proAiButton")

        self.ai_difficulty_layout.addWidget(self.easy_ai_button)
        self.ai_difficulty_layout.addWidget(self.hard_ai_button)
        self.ai_difficulty_layout.addWidget(self.pro_ai_button)

        self.right_panel_layout.addWidget(self.ai_difficulty_label)
        self.right_panel_layout.addLayout(self.ai_difficulty_layout)

        # Set initial AI difficulty button style
        self.easy_ai_button.setStyleSheet(self.get_gradient_button_style("#4CAF50", "#28a745")) # Default easy

        # Removed Volume Controls from game UI

        # Game control buttons
        self.control_layout = QHBoxLayout()
        self.new_game_button = QPushButton("New Game", self)
        self.new_game_button.setObjectName("newGameButton")
        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setObjectName("quitButton")
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setObjectName("undoButton")
        self.undo_button.setEnabled(False) # Disabled by default
        self.redo_button = QPushButton("Redo", self) # New: Redo button
        self.redo_button.setObjectName("redoButton")
        self.redo_button.setEnabled(False) # Disabled by default

        self.control_layout.addWidget(self.new_game_button)
        self.control_layout.addWidget(self.undo_button) # Undo next to New Game
        self.control_layout.addWidget(self.redo_button) # New: Redo button
        self.right_panel_layout.addLayout(self.control_layout)

        # Digital Clock (Moved here)
        self.clock_label = QLabel()
        self.clock_label.setObjectName("clockLabel")
        self.clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock_label.setFont(QFont("Consolas", 18, QFont.Bold)) # A common monospace font
        self.right_panel_layout.addWidget(self.clock_label) # Add clock below control buttons

        # Clock Timer
        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000) # Update every 1 second
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start()
        self.update_clock() # Initial call to set the time immediately

        self.right_panel_layout.addStretch(1) # Push controls to top

        # Add quit button at the very bottom of the right panel
        self.right_panel_layout.addWidget(self.quit_button)


        # --- Button Connections ---
        # Board style buttons are now handled by QMenu actions
        self.new_game_button.clicked.connect(self.reset_game)
        self.quit_button.clicked.connect(self.close)
        self.undo_button.clicked.connect(self.handle_undo)
        self.redo_button.clicked.connect(self.handle_redo) # New: Redo button connection

        # AI difficulty button connections
        self.easy_ai_button.clicked.connect(lambda: self.set_ai_difficulty('easy'))
        self.hard_ai_button.clicked.connect(lambda: self.set_ai_difficulty('hard'))
        self.pro_ai_button.clicked.connect(lambda: self.set_ai_difficulty('pro'))

        self.media_player.setVolume(self.global_volume) # Set initial volume from global setting

        # Initially hide AI difficulty and volume controls if not in AI mode
        if self.game_mode == "two_player":
            self.ai_difficulty_label.hide()
            self.easy_ai_button.hide()
            self.hard_ai_button.hide()
            self.pro_ai_button.hide()


        # --- Mouse Interaction Setup ---
        self.selected_piece_item = None # QGraphicsPixmapItem of the selected piece
        self.selected_pos = None # (row, col) of the selected piece
        self.view.mousePressEvent = self.handle_square_click

        # --- AI Timer ---
        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self.trigger_ai_move) # Renamed for clarity

    def play_move_sound(self):
        """Plays the chess piece move sound."""
        if self.move_sound_url and self.move_sound_url.isValid() and os.path.exists(self.move_sound_url.toLocalFile()):
            self.media_player.setMedia(QMediaContent(self.move_sound_url))
            self.media_player.play()
        else:
            print("Move sound file not available or invalid. Cannot play sound.")

    def play_capture_sound(self):
        """Plays the chess piece capture sound."""
        if self.capture_sound_url and self.capture_sound_url.isValid() and os.path.exists(self.capture_sound_url.toLocalFile()):
            self.media_player.setMedia(QMediaContent(self.capture_sound_url))
            self.media_player.play()
        else:
            print("Capture sound file not available or invalid. Cannot play sound.")

    def play_check_sound(self):
        """Plays the check sound."""
        if self.check_sound_url and self.check_sound_url.isValid() and os.path.exists(self.check_sound_url.toLocalFile()):
            self.media_player.setMedia(QMediaContent(self.check_sound_url))
            self.media_player.play()
        else:
            print("Check sound file not available or invalid. Cannot play sound.")

    def update_clock(self):
        """Updates the digital clock label with the current time."""
        current_time = datetime.now().strftime("%H:%M:%S") # Format as HH:MM:SS
        self.clock_label.setText(current_time)


    def get_gradient_button_style(self, color1: str, color2: str) -> str:
        """
        Generates a QSS style string for a QPushButton with a linear gradient background.
        """
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:1 {color2});
                color: white;
                padding: 12px 18px;
                border: none;
                border-radius: 8px; /* Default rounded corners for all QPushButtons */
                font-size: 13px;
                font-weight: bold;
                margin: 4px;
                min-width: 80px;
                transition: background 0.3s ease, transform 0.2s ease;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color2}, stop:1 {color1});
                transform: translateY(-2px);
            }}
            QPushButton:pressed {{
                background-color: {color1}; /* Solid color on press for immediate feedback */
                transform: translateY(0);
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
                transform: none;
            }}
        """

    def get_embedded_qss(self) -> str:
        """
        Returns the QSS (Qt Style Sheet) as a multi-line string.
        This defines the visual appearance of various GUI elements.
        """
        return f"""
/* General Window Styling */
QMainWindow {{
    background-color: #f0f4f8; /* Light blue-grey background for the main window */
    color: #2c3e50; /* Dark text color for contrast */
    font-family: "Cinzel", sans-serif; /* Modern font */
    font-size: 14px;
}}

/* All PushButtons - Base style with gradient */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9); /* Default blue gradient */
    color: white;
    padding: 12px 18px;
    border: none;
    border-radius: 8px; /* Rounded corners for all buttons by default */
    font-size: 13px;
    font-weight: bold;
    margin: 4px;
    min-width: 80px;
    transition: background 0.3s ease, transform 0.2s ease;
}}

QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2980b9, stop:1 #3498db); /* Reverse gradient on hover */
    transform: translateY(-2px);
}}

QPushButton:pressed {{
    background-color: #2471a3; /* Solid color on press */
    transform: translateY(0);
}}

QPushButton:disabled {{
    background-color: #cccccc;
    color: #666666;
    transform: none;
}}

/* Specific Button Styles (using objectName) - Override default if needed */
QPushButton#boardStylesMenuButton {{ {self.get_gradient_button_style("#007bff", "#0056b3")} }}
QPushButton#pieceStylesMenuButton {{ {self.get_gradient_button_style("#007bff", "#0056b3")} }}
QPushButton#returnHomeButton {{ {self.get_gradient_button_style("#6c757d", "#5a6268")} }}


QPushButton#newGameButton {{ {self.get_gradient_button_style("#6f42c1", "#563d7c")} }}
QPushButton#quitButton {{ {self.get_gradient_button_style("#dc3545", "#c82333")} }} /* Red gradient for quit */
QPushButton#undoButton {{ {self.get_gradient_button_style("#ffc107", "#e0a800")} }} /* Orange/Yellow for undo */
QPushButton#redoButton {{ {self.get_gradient_button_style("#17a2b8", "#138496")} }} /* Cyan for redo */

/* Volume Control Buttons (now used in SettingsDialog) */
QPushButton#volumeUpButton, QPushButton#volumeDownButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #20c997, stop:1 #17a2b8); /* Teal/Green gradient */
    color: white;
    padding: 5px 10px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 8px;
    min-width: 30px;
    max-width: 40px;
}}
QPushButton#volumeUpButton:hover, QPushButton#volumeDownButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #17a2b8, stop:1 #20c997);
}}


/* AI Difficulty Buttons - Smaller size, specific gradient */
QPushButton#easyAiButton, QPushButton#hardAiButton, QPushButton#proAiButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #0056b3); /* Default blue gradient */
    color: white;
    padding: 8px 12px;
    font-size: 12px;
    min-width: 60px;
    border-radius: 8px;
}}
QPushButton#easyAiButton:hover, QPushButton#hardAiButton:hover, QPushButton#proAiButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0056b3, stop:1 #007bff);
}}

/* Style for the CURRENTLY SELECTED AI Difficulty Button */
QPushButton#easyAiButton[style*="background-color: lightgreen;"],
QPushButton#hardAiButton[style*="background-color: lightgreen;"],
QPushButton#proAiButton[style*="background-color: lightgreen;"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #28a745); /* Green gradient for selected */
    color: white;
    border: 2px solid #28a745;
    font-weight: bold;
}}


/* All Labels */
QLabel {{
    color: #333333;
    font-size: 15px;
    padding: 3px;
}}

QLabel#statusLabel {{
    font-size: 25px;
    font-weight: bold;
    color: #000000;
    margin-bottom: 8px;
    padding: 8px;
    background-color: #e0e0e0;
    border-radius: 5px;
    border: 1px solid #c0c0c0;
}}

QLabel#capturedLabel {{
    font-size: 13px;
    color: #555555;
    min-height: 25px;
    border-bottom: 1px solid #dddddd;
    padding-bottom: 5px;
    margin-bottom: 5px;
    font-weight: bold;
}}

QLabel#aiDifficultyLabel, QLabel#volumeLabel {{
    font-size: 15px;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #444444;
}}

QLabel#clockLabel {{
    font-size: 20px;
    font-weight: bold;
    color: #333;
    background-color: #f8f8f8;
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 5px 10px;
    min-width: 100px;
}}

/* Frames */
QFrame#capturedFrame {{
    background-color: #ffffff;
    border: 1px solid #aaaaaa;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 15px;
}}

/* Graphics View (Chess Board Display Area) */
QGraphicsView {{
    border: 2px solid #777777;
    background-color: #eeeeee;
    border-radius: 5px;
}}

/* Pawn Promotion Dialog */
QDialog {{
    background-color: #f8f8f8;
    color: #333333;
    border: 1px solid #cccccc;
    border-radius: 8px;
}}

QDialog QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #0056b3);
    color: white;
    padding: 10px 18px;
    border-radius: 8px; /* Rounded corners */
    font-size: 13px;
    font-weight: bold;
}}
QDialog QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0056b3, stop:1 #007bff);
}}

/* QMenu styling */
QMenu {{
    background-color: #f8f8f8;
    border: 1px solid #cccccc;
    border-radius: 5px;
    padding: 5px;
}}

QMenu::item {{
    padding: 8px 20px;
    background-color: transparent;
    color: #333333;
}}

QMenu::item:selected {{
    background-color: #007bff;
    color: white;
    border-radius: 3px;
}}

QMenu::separator {{
    height: 1px;
    background: #dddddd;
    margin: 5px 0px;
}}

/* QSlider Styling */
QSlider::groove:horizontal {{
    border: 1px solid #bbb;
    background: #d3d3d3;
    height: 8px;
    border-radius: 4px;
}}

QSlider::sub-page:horizontal {{
    background: #3498db;
    border: 1px solid #777;
    height: 8px;
    border-radius: 4px;
}}

QSlider::add-page:horizontal {{
    background: #fff;
    border: 1px solid #777;
    height: 8px;
    border-radius: 4px;
}}

QSlider::handle:horizontal {{
    background: #3498db;
    border: 1px solid #3498db;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}}

QSlider::handle:horizontal:hover {{
    background: #2980b9;
    border: 1px solid #2980b9;
}}
        """

    def setup_piece_images(self, image_directory: str):
        """
        Loads chess piece images from the specified local directory.
        Sets a flag (`use_unicode_pieces`) if image loading fails, to fall back to unicode.
        """
        self.piece_images = {}
        # Unicode representations for chess pieces (fallback)
        self.unicode_pieces = {
            'wK': '♔', 'wQ': '♕', 'wR': '♖', 'wB': '♗', 'wN': '♘', 'wP': '♙',
            'bK': '♚', 'bQ': '♛', 'bR': '♜', 'bB': '♝', 'bN': '♞', 'bP': '♟'
        }

        # List of all piece codes to load
        piece_codes = ['wK', 'wQ', 'wR', 'wB', 'wN', 'wP',
                       'bK', 'bQ', 'bR', 'bP', 'bN', 'bB']

        all_images_found = True
        for piece_code in piece_codes:
            # Attempt to download or locate the image
            img_path = download_image_if_not_exists(
                f"{piece_code}.png",
                image_directory # Use the passed directory here
            )
            if img_path:
                self.piece_images[piece_code] = img_path
            else:
                all_images_found = False
                print(f"Could not find image for {piece_code} in {image_directory}. Will use unicode.")
                # If image not found, store None or a placeholder to indicate unicode fallback
                self.piece_images[piece_code] = None

        if not all_images_found:
            self.use_unicode_pieces = True
            QMessageBox.warning(self, "Image Warning",
                                "Some or all chess piece images could not be loaded from the selected set. "
                                "Unicode chess symbols will be used instead. Please ensure image files are in the correct directory.")
        else:
            self.use_unicode_pieces = False # Reset if all images are found for the new set

    def set_piece_set(self, directory: str):
        """
        Switches the active piece image set and refreshes the board.
        """
        self.current_piece_set_dir = directory
        self.setup_piece_images(self.current_piece_set_dir)
        self.refresh_board()

    def set_ai_difficulty_buttons_enabled(self, enabled: bool):
        """
        Enables or disables the AI difficulty selection buttons and their label.
        This method is now largely redundant as visibility is handled by set_game_mode.
        Kept for potential future granular control.

        Args:
            enabled (bool): True to enable, False to disable.
        """
        self.easy_ai_button.setEnabled(enabled)
        self.hard_ai_button.setEnabled(enabled)
        self.pro_ai_button.setEnabled(enabled)
        self.ai_difficulty_label.setEnabled(enabled)

    def set_ai_difficulty(self, difficulty: str):
        """
        Sets the AI difficulty level in the ChessLogic and updates the UI button styles.

        Args:
            difficulty (str): 'easy', 'hard', or 'pro'.
        """
        self.chess_logic.ai_difficulty = difficulty

        # Reset all AI difficulty button styles to default gradient
        self.easy_ai_button.setStyleSheet(self.get_gradient_button_style("#007bff", "#0056b3"))
        self.hard_ai_button.setStyleSheet(self.get_gradient_button_style("#007bff", "#0056b3"))
        self.pro_ai_button.setStyleSheet(self.get_gradient_button_style("#007bff", "#0056b3"))

        # Apply highlight style (green gradient) to the newly selected button
        if difficulty == 'easy':
            self.easy_ai_button.setStyleSheet(self.get_gradient_button_style("#4CAF50", "#28a745"))
        elif difficulty == 'hard':
            self.hard_ai_button.setStyleSheet(self.get_gradient_button_style("#4CAF50", "#28a745"))
        elif difficulty == 'pro':
            self.pro_ai_button.setStyleSheet(self.get_gradient_button_style("#4CAF50", "#28a745"))

        # Update status label to reflect new AI difficulty
        self.status_label.setText(f"Playing against AI ({self.chess_logic.ai_difficulty.capitalize()}) - White's turn")
        self.reset_game() # Reset game with new AI difficulty setting

    def reset_game(self):
        """
        Resets the game to its initial state, clearing the board, resetting game flags,
        and updating the display.
        """
        # Store current AI difficulty before creating new ChessLogic instance
        current_ai_difficulty = self.chess_logic.ai_difficulty

        self.chess_logic = ChessLogic() # Create a new game logic instance
        self.chess_logic.ai_difficulty = current_ai_difficulty # Apply the stored AI difficulty

        self.selected_pos = None
        self.valid_moves = []

        # Re-setup piece images for the current set
        self.setup_piece_images(self.current_piece_set_dir) # Ensure correct piece set is loaded
        self.refresh_board() # Re-draw board and pieces
        self.update_captured_pieces_display() # Clear captured pieces display

        # Update status label based on current game mode
        if self.game_mode == "two_player":
            self.status_label.setText("Two Player Mode - White's turn")
            self.undo_button.hide() # Ensure undo is hidden
            self.redo_button.hide() # Ensure redo is hidden
        else:
            self.status_label.setText(f"Playing against AI ({self.chess_logic.ai_difficulty.capitalize()}) - White's turn")
            self.undo_button.show() # Ensure undo is shown
            self.redo_button.show() # Ensure redo is shown

        self.undo_button.setEnabled(len(self.chess_logic.move_history) > 0) # Reset enabled state
        self.redo_button.setEnabled(len(self.chess_logic.redo_history) > 0) # Reset enabled state


    def create_standard_chessboard(self):
        """
        Creates the chessboard with standard white and gray squares.
        """
        self.scene.clear() # Clear all items from the scene
        self._draw_chessboard_squares(QColor(255, 255, 255), QColor(169, 169, 169))
        self.place_pieces()

    def create_styled_chessboard1(self):
        """
        Creates a chessboard with a vintage/sepia color scheme.
        """
        self.scene.clear()
        self._draw_chessboard_squares(QColor(255, 223, 186), QColor(111, 85, 59))
        self.place_pieces()

    def create_styled_chessboard2(self):
        """
        Creates a chessboard with a dark/modern color scheme.
        """
        self.scene.clear()
        self._draw_chessboard_squares(QColor(150, 150, 150), QColor(50, 50, 50))
        self.place_pieces()

    def create_alternative_chessboard(self):
        """
        Creates a chessboard with a high-contrast color scheme.
        """
        self.scene.clear()
        self._draw_chessboard_squares(QColor(220, 220, 220), QColor(40, 40, 40))
        self.place_pieces()

    def show_custom_color_dialog(self):
        """
        Opens color dialogs for light and dark squares and applies custom colors.
        """
        # Get current colors to pre-select in dialogs
        current_light_color = self.current_light_square_color
        current_dark_color = self.current_dark_square_color

        # Dialog for light squares
        light_color = QColorDialog.getColor(current_light_color, self, "Choose Light Square Color")
        if not light_color.isValid():
            return # User cancelled

        # Dialog for dark squares
        dark_color = QColorDialog.getColor(current_dark_color, self, "Choose Dark Square Color")
        if not dark_color.isValid():
            return # User cancelled

        self._draw_chessboard_squares(light_color, dark_color)
        self.place_pieces() # Re-place pieces on new board colors

    def _draw_chessboard_squares(self, light_color: QColor, dark_color: QColor):
        """
        Helper method to draw the 64 squares of the chessboard with specified light and dark colors.
        Stores the colors as instance variables.
        """
        self.current_light_square_color = light_color
        self.current_dark_square_color = dark_color

        for row in range(self.board_height):
            for col in range(self.board_width):
                x = col * self.square_size
                y = row * self.square_size
                square_color = light_color if (row + col) % 2 == 0 else dark_color
                rect = QGraphicsRectItem(QRectF(x, y, self.square_size, self.square_size))
                rect.setBrush(square_color)
                self.scene.addItem(rect)

    def place_pieces(self):
        """
        Places or updates the chess pieces on the board according to the current game state
        from `self.chess_logic.board`. Uses images if available, otherwise unicode symbols.
        """
        # Remove existing piece items before redrawing
        # This loop is now redundant as scene.clear() is called before _draw_chessboard_squares
        # which clears all items including pieces.
        # However, if place_pieces is called independently of _draw_chessboard_squares,
        # this loop would be necessary. For safety, I'll keep it but note its redundancy
        # if always called after scene.clear().
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem) or isinstance(item, QGraphicsTextItem):
                self.scene.removeItem(item)

        for row in range(self.board_height):
            for col in range(self.board_width):
                piece = self.chess_logic.get_piece(row, col)
                if piece:
                    if self.use_unicode_pieces:
                        # Fallback to unicode characters if images are not available
                        text_item = self.scene.addText(self.unicode_pieces.get(piece, '?'))
                        text_item.setFont(QFont("Arial Unicode MS", int(self.square_size * 0.6)))
                        # Position unicode text to be roughly centered
                        text_item.setPos(col * self.square_size + self.square_size * 0.15,
                                         row * self.square_size + self.square_size * 0.05)
                        # Adjust color for better visibility on different square colors
                        # Unicode pieces should be visible on both light and dark squares.
                        # A simple approach is to make them black on light squares and white on dark squares.
                        if (row + col) % 2 == 0: # Light square
                            text_item.setDefaultTextColor(Qt.black)
                        else: # Dark square
                            text_item.setDefaultTextColor(Qt.white)
                    else:
                        # Use image files for pieces
                        try:
                            piece_image_path = self.piece_images.get(piece)
                            if piece_image_path:
                                piece_image = QPixmap(piece_image_path)
                                # Scale the image to fit the square with some padding
                                piece_image = piece_image.scaled(
                                    int(self.square_size * 0.9),
                                    int(self.square_size * 0.9),
                                    Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation
                                )
                                piece_item = QGraphicsPixmapItem(piece_image)
                                # Center the image within its square
                                offset_x = col * self.square_size + (self.square_size - piece_image.width()) / 2
                                offset_y = row * self.square_size + (self.square_size - piece_image.height()) / 2
                                piece_item.setPos(offset_x, offset_y) # Use setPos for QGraphicsPixmapItem
                                self.scene.addItem(piece_item)
                            else:
                                # Fallback to unicode if a specific image is missing even if use_unicode_pieces is False
                                print(f"Image for {piece} not found, falling back to unicode.")
                                text_item = self.scene.addText(self.unicode_pieces.get(piece, '?'))
                                text_item.setFont(QFont("Arial Unicode MS", int(self.square_size * 0.6)))
                                text_item.setPos(col * self.square_size + self.square_size * 0.15,
                                                 row * self.square_size + self.square_size * 0.05)
                                if (row + col) % 2 == 0: # Light square
                                    text_item.setDefaultTextColor(Qt.black)
                                else: # Dark square
                                    text_item.setDefaultTextColor(Qt.white)
                        except Exception as e:
                            print(f"Error loading piece image for {piece}: {e}. Falling back to unicode.")
                            # Ensure unicode fallback even if an exception occurs during image loading
                            text_item = self.scene.addText(self.unicode_pieces.get(piece, '?'))
                            text_item.setFont(QFont("Arial Unicode MS", int(self.square_size * 0.6)))
                            text_item.setPos(col * self.square_size + self.square_size * 0.15,
                                             row * self.square_size + self.square_size * 0.05)
                            if (row + col) % 2 == 0: # Light square
                                text_item.setDefaultTextColor(Qt.black)
                            else: # Dark square
                                text_item.setDefaultTextColor(Qt.white)

    def draw_move_highlight(self, valid_moves: list[tuple[int, int]]):
        """
        Draws green dots on the board to visually indicate valid squares for the selected piece.

        Args:
            valid_moves (list[tuple[int, int]]): A list of (row, col) tuples representing valid moves.
        """
        for (r, c) in valid_moves:
            x = c * self.square_size
            y = r * self.square_size
            # Draw a small green ellipse in the center of the square
            ellipse = QGraphicsEllipseItem(QRectF(x + self.square_size/3, y + self.square_size/3, self.square_size/3, self.square_size/3))
            ellipse.setBrush(QColor(0, 200, 0, 200))  # Green dot with transparency
            ellipse.setZValue(1) # Ensure highlights are drawn on top of squares and pieces
            self.scene.addItem(ellipse)

    def handle_square_click(self, event):
        """
        Handles mouse click events on the chessboard.
        Manages piece selection, move validation, and execution.
        """
        # Ignore clicks if game is over or AI is thinking
        if self.chess_logic.game_over or self.ai_thinking:
            return

        mouse_pos = event.pos()
        scene_pos = self.view.mapToScene(mouse_pos)
        row = int(scene_pos.y() // self.square_size)
        col = int(scene_pos.x() // self.square_size)

        # Ensure click is within board boundaries
        if not (0 <= row < self.board_height and 0 <= col < self.board_width):
            print("Clicked outside the board.")
            return

        clicked_piece = self.chess_logic.get_piece(row, col)

        if self.selected_pos is None:
            # No piece currently selected, try to select one
            if clicked_piece and clicked_piece[0] == self.chess_logic.current_turn:
                self.selected_pos = (row, col)
                # Refresh board to clear any old highlights and then draw new ones
                self.refresh_board()
                self.valid_moves = self.chess_logic.highlight_moves((row, col))
                self.draw_move_highlight(self.valid_moves)
            else:
                # Clicked on an empty square or opponent's piece when nothing selected
                pass # Do nothing, wait for a valid selection
        else:
            # A piece is already selected, this click is a potential move destination
            start_pos = self.selected_pos
            end_pos = (row, col)
            selected_piece_on_board = self.chess_logic.get_piece(start_pos[0], start_pos[1])

            if not selected_piece_on_board: # Defensive check
                self.selected_pos = None
                self.valid_moves = []
                self.refresh_board()
                return

            if end_pos in self.valid_moves:
                # Valid move, execute it
                
                # Determine if it's a capture
                is_capture = self.chess_logic.get_piece(end_pos[0], end_pos[1]) is not None

                # Store move details for undo functionality BEFORE making the move
                move_info = {
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'moved_piece': selected_piece_on_board,
                    'captured_piece': self.chess_logic.get_piece(end_pos[0], end_pos[1]), # Piece at end_pos before move
                    'promoted_to': None, # Will be updated if promotion occurs
                    'castling_info': None, # Will be updated if castling occurs
                    'en_passant_target_before_move': self.chess_logic.en_passant_target, # Store current EP target
                    'is_en_passant_capture': False # Will be updated if EP capture occurs
                }

                # Handle en passant capture for the actual move
                if selected_piece_on_board[1] == 'P' and end_pos == self.chess_logic.en_passant_target:
                    # If it's an en passant capture, the captured pawn is not at end_pos
                    captured_pawn_row = start_pos[0]
                    captured_pawn_col = end_pos[1]
                    captured_piece_ep = self.chess_logic.get_piece(captured_pawn_row, captured_pawn_col)
                    if captured_piece_ep:
                        self.chess_logic.captured_pieces[selected_piece_on_board[0]].append(captured_piece_ep)
                        self.chess_logic.place_piece(captured_pawn_row, captured_pawn_col, None) # Remove captured pawn
                        move_info['is_en_passant_capture'] = True
                        move_info['captured_piece'] = captured_piece_ep # Store the actual captured pawn for undo
                    is_capture = True # En passant is also a capture

                # Handle regular capture
                elif move_info['captured_piece']:
                    self.chess_logic.captured_pieces[selected_piece_on_board[0]].append(move_info['captured_piece'])
                    is_capture = True

                self.update_captured_pieces_display() # Update display after capture

                # Play sound based on whether it was a capture or a regular move
                if is_capture:
                    self.play_capture_sound()
                else:
                    self.play_move_sound()


                # Handle king and rook movement for castling
                if selected_piece_on_board[1] == 'K':
                    self.chess_logic.kings_moved[selected_piece_on_board[0]] = True
                    if abs(start_pos[1] - end_pos[1]) == 2: # Castling move
                        # Determine if kingside or queenside castling
                        if end_pos[1] > start_pos[1]: # Kingside
                            rook_start_col = 7
                            rook_end_col = end_pos[1] - 1
                        else: # Queenside
                            rook_start_col = 0
                            rook_end_col = end_pos[1] + 1

                        rook = self.chess_logic.get_piece(start_pos[0], rook_start_col)
                        self.chess_logic.place_piece(start_pos[0], rook_end_col, rook) # Move rook
                        self.chess_logic.place_piece(start_pos[0], rook_start_col, None) # Clear old rook square
                        move_info['castling_info'] = {'rook_start': (start_pos[0], rook_start_col), 'rook_end': (start_pos[0], rook_end_col), 'rook_piece': rook}

                # Track rook movement for castling
                elif selected_piece_on_board[1] == 'R':
                    color = selected_piece_on_board[0]
                    if start_pos[0] == (7 if color == 'w' else 0):
                        if start_pos[1] == 0:  # Queenside rook
                            self.chess_logic.rooks_moved[color]['queenside'] = True # Corrected access
                        elif start_pos[1] == 7:  # Kingside rook
                            self.chess_logic.rooks_moved[color]['kingside'] = True # Corrected access

                # Make the actual piece move on the board
                self.chess_logic.place_piece(end_pos[0], end_pos[1], selected_piece_on_board)
                self.chess_logic.place_piece(start_pos[0], start_pos[1], None)

                # Check for pawn promotion
                if self.chess_logic.handle_pawn_promotion(end_pos[0], end_pos[1]):
                    self.handle_promotion_dialog(end_pos[0], end_pos[1], selected_piece_on_board[0])
                    move_info['promoted_to'] = self.chess_logic.get_piece(end_pos[0], end_pos[1])[1] # Store promoted piece type

                # Update en passant target for the next turn
                self.chess_logic.en_passant_target = None
                if selected_piece_on_board[1] == 'P' and abs(start_pos[0] - end_pos[0]) == 2:
                    ep_row = (start_pos[0] + end_pos[0]) // 2
                    self.chess_logic.en_passant_target = (ep_row, start_pos[1])

                self.chess_logic.move_history.append(move_info) # Add completed move to history
                self.chess_logic.redo_history.clear() # Clear redo history when a new move is made

                # Change turn
                self.chess_logic.current_turn = 'b' if self.chess_logic.current_turn == 'w' else 'w'

                # Reset selection and refresh board
                self.selected_pos = None
                self.valid_moves = []
                self.refresh_board()

                # Check game status after the move
                self.check_game_status()

                # If playing against AI and it's AI's turn, trigger AI move
                if self.game_mode == "vs_ai" and self.chess_logic.current_turn == 'b' and not self.chess_logic.game_over:
                    self.ai_thinking = True
                    self.status_label.setText("AI is thinking...")
                    self.ai_timer.start(469)  # Small delay to simulate thinking and allow UI update
            else:
                # Clicked on an invalid square or own piece (re-select)
                if clicked_piece and clicked_piece[0] == self.chess_logic.current_turn:
                    self.selected_pos = (row, col)
                    self.refresh_board() # Clear old highlights
                    self.valid_moves = self.chess_logic.highlight_moves((row, col))
                    self.draw_move_highlight(self.valid_moves)
                else:
                    # Deselect if clicking elsewhere (empty square or opponent's piece)
                    self.selected_pos = None
                    self.valid_moves = []
                    self.refresh_board() # Clear all highlights

    def trigger_ai_move(self):
        """
        Triggers the AI to make its move after a short delay.
        This method is connected to the QTimer.
        """
        self.ai_timer.stop() # Stop the timer once triggered

        if self.chess_logic.make_ai_move():
            # Check if the AI's move resulted in a capture
            last_move_info = self.chess_logic.move_history[-1] if self.chess_logic.move_history else None
            if last_move_info and (last_move_info['captured_piece'] or last_move_info.get('is_en_passant_capture')):
                self.play_capture_sound()
            else:
                self.play_move_sound() # Play sound after AI move

            # AI successfully made a move, now change turn back to player
            self.chess_logic.current_turn = 'w'
            self.refresh_board()
            self.update_captured_pieces_display() # Update display after AI capture

            # Check game status after AI's move
            self.check_game_status()

        self.ai_thinking = False # AI finished thinking, allow user input
        self.undo_button.setEnabled(len(self.chess_logic.move_history) > 0)
        self.redo_button.setEnabled(len(self.chess_logic.redo_history) > 0)


    def handle_undo(self):
        """
        Handles the undo button click. In 'vs AI' mode, it attempts to undo
        both the player's last move and the AI's last move.
        """
        if self.game_mode == "vs_ai":
            # Undo player's move
            if self.chess_logic.undo_last_move():
                self.refresh_board()
                self.update_captured_pieces_display()
                self.check_game_status() # Update status after undo

            # Undo AI's move (if AI has made a move and there's history)
            if self.chess_logic.undo_last_move():
                self.refresh_board()
                self.update_captured_pieces_display()
                self.check_game_status() # Update status after undo
            else:
                QMessageBox.information(self, "Undo", "No more moves to undo.")
        else:
            QMessageBox.information(self, "Undo", "Undo is only available in 'vs AI' mode.")

        # Update button states after undo attempt
        self.undo_button.setEnabled(len(self.chess_logic.move_history) > 0)
        self.redo_button.setEnabled(len(self.chess_logic.redo_history) > 0)

    def handle_redo(self):
        """
        Handles the redo button click. In 'vs AI' mode, it attempts to redo
        both the AI's last move and the player's last move.
        """
        if self.game_mode == "vs_ai":
            # Redo AI's move
            if self.chess_logic.redo_last_move():
                self.refresh_board()
                self.update_captured_pieces_display()
                self.check_game_status()

            # Redo player's move (if available)
            if self.chess_logic.redo_last_move():
                self.refresh_board()
                self.update_captured_pieces_display()
                self.check_game_status()
            else:
                QMessageBox.information(self, "Redo", "No more moves to redo.")
        else:
            QMessageBox.information(self, "Redo", "Redo is only available in 'vs AI' mode.")

        # Update button states after redo attempt
        self.undo_button.setEnabled(len(self.chess_logic.move_history) > 0)
        self.redo_button.setEnabled(len(self.chess_logic.redo_history) > 0)


    def handle_promotion_dialog(self, row: int, col: int, color: str):
        """
        Shows the pawn promotion dialog and applies the chosen promotion.

        Args:
            row (int): The row of the pawn to be promoted.
            col (int): The column of the pawn to be promoted.
            color (str): The color of the pawn ('w' or 'b').
        """
        # Pass the current ChessBoard instance as parent, so PawnPromotionDialog can access current_piece_set_dir
        dialog = PawnPromotionDialog(color, self)
        if dialog.exec_(): # Show dialog modally
            promoted_piece_type = dialog.selected_piece
            self.chess_logic.promote_pawn(row, col, promoted_piece_type)
        else:
            # If dialog is cancelled or closed, default to Queen promotion
            self.chess_logic.promote_pawn(row, col, 'Q')

    def check_game_status(self):
        """
        Checks the current game state for check, checkmate, or stalemate
        and updates the status label and displays appropriate messages.
        """
        current_color = self.chess_logic.current_turn
        opponent_color = 'w' if current_color == 'b' else 'b' # Corrected opponent_color logic

        # Check if the current player (whose turn it is) is in check
        in_check = self.chess_logic.is_opponent_in_check(current_color)

        # Check for checkmate
        if in_check and self.chess_logic.is_checkmate(current_color):
            self.chess_logic.game_over = True
            self.chess_logic.winner = opponent_color
            winner_name = "White" if opponent_color == 'w' else "Black" # Corrected winner_name assignment
            self.status_label.setText(f"Checkmate! {winner_name} wins!")
            QMessageBox.information(self, "Game Over", f"Checkmate! {winner_name} wins! 👑 Congratulations! {winner_name}", QMessageBox.Ok, QMessageBox.Ok)
            # Show New Game button in the message box
            self.new_game_button.show() # Ensure it's visible if hidden

            # Update window icon to logo upon game end
            if self.logo_pixmap:
                self.setWindowIcon(QIcon(self.logo_pixmap))
            return

        # Check for stalemate
        if self.chess_logic.is_stalemate(current_color):
            self.chess_logic.game_over = True
            self.status_label.setText("Stalemate! The game is a draw.")
            QMessageBox.information(self, "Game Over", "Stalemate! 🤝 The game is a draw.", QMessageBox.Ok, QMessageBox.Ok)
            # Show New Game button in the message box
            self.new_game_button.show() # Ensure it's visible if hidden

            # Update window icon to logo upon game end
            if self.logo_pixmap:
                self.setWindowIcon(QIcon(self.logo_pixmap))
            return

        # Update status for check or normal turn
        if in_check:
            turn_text = "White's" if current_color == 'w' else "Black's"
            self.status_label.setText(f"{turn_text} turn - Check!")
            self.play_check_sound() # Play check sound
        else:
            turn_text = "White's" if current_color == 'w' else "Black's"
            self.status_label.setText(f"{turn_text} turn")

        # Update button enabled states
        self.undo_button.setEnabled(len(self.chess_logic.move_history) > 0)
        self.redo_button.setEnabled(len(self.chess_logic.redo_history) > 0)


    def refresh_board(self):
        """
        Refreshes the entire board display, redrawing squares and pieces.
        This is called after every move or when changing board styles.
        """
        self.scene.clear()
        self._draw_chessboard_squares(self.current_light_square_color, self.current_dark_square_color)
        self.place_pieces()

    def update_captured_pieces_display(self):
        """
        Updates the visual display of captured pieces for both white and black.
        Removes old icons and adds new ones based on the `chess_logic.captured_pieces` lists.
        """
        # Clear existing captured piece images from the display layouts
        # Iterate in reverse to safely remove widgets from layout
        for i in reversed(range(self.white_caps_display.count())):
            widget_to_remove = self.white_caps_display.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None) # Remove from layout and delete widget

        for i in reversed(range(self.black_caps_display.count())):
            widget_to_remove = self.black_caps_display.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None) # Remove from layout and delete widget

        # Add captured white pieces (captured by black) to black's captured display
        for piece_code in self.chess_logic.captured_pieces['b']: # These are white pieces captured by black
            self._add_captured_piece_icon(piece_code, self.black_caps_display, 'black')

        # Add captured black pieces (captured by white) to white's captured display
        for piece_code in self.chess_logic.captured_pieces['w']: # These are black pieces captured by white
            self._add_captured_piece_icon(piece_code, self.white_caps_display, 'white')

    def _add_captured_piece_icon(self, piece_code: str, layout: QHBoxLayout, text_color: str):
        """
        Helper method to add a single captured piece icon (image or unicode) to a layout.

        Args:
            piece_code (str): The string code of the piece (e.g., 'wP', 'bN').
            layout (QHBoxLayout): The layout to add the icon to.
            text_color (str): 'white' or 'black' for unicode piece color.
        """
        if self.use_unicode_pieces:
            # Use unicode characters for captured pieces
            label = QLabel(self.unicode_pieces.get(piece_code, '?'))
            label.setFont(QFont("Arial Unicode MS", 16))
            label.setStyleSheet(f"color: {text_color};") # Set color for unicode
            layout.addWidget(label)
        else:
            # Use image files for captured pieces
            try:
                pixmap_path = self.piece_images.get(piece_code)
                if pixmap_path:
                    pixmap = QPixmap(pixmap_path)
                    # Scale captured piece images to a smaller size
                    pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    label = QLabel()
                    label.setPixmap(pixmap)
                    layout.addWidget(label)
                else:
                    # Fallback to unicode if a specific image is missing
                    print(f"Image for captured piece {piece_code} not found, falling back to unicode.")
                    label = QLabel(self.unicode_pieces.get(piece_code, '?'))
                    label.setFont(QFont("Arial Unicode MS", 16))
                    label.setStyleSheet(f"color: {text_color};")
                    layout.addWidget(label)
            except Exception as e:
                print(f"Error loading captured piece image {piece_code}: {e}. Falling back to unicode.")
                # Ensure unicode fallback even if an exception occurs
                label = QLabel(self.unicode_pieces.get(piece_code, '?'))
                label.setFont(QFont("Arial Unicode MS", 16))
                label.setStyleSheet(f"color: {text_color};")
                layout.addWidget(label)


# --- Main Application Entry Point ---
if __name__ == "__main__":
    # Create the QApplication instance
    app = QApplication(sys.argv)

    # Initialize the core chess game logic
    chess_logic = ChessLogic()

    # Create and show the main ChessBoard GUI window
    window = ChessBoard(chess_logic)
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec_())

    # These print statements will only execute after the application closes
    print("Chess application closed.")
    print("Thank you for playing! ")
    print("Goodbye!")

# Note: The print statements at the end will only execute after the application closes.
# This is a chess application built with PyQt5, featuring a customizable chessboard,
# AI opponent, and support for different piece sets and styles. It includes features like
# undo, redo, promotion, and more.
# The code is structured to allow easy customization and extension for future features.
# Enjoy playing chess! ♟️

import numpy as np
from enum import Enum
from typing import Optional
from typing import Callable, Tuple
from numba import njit


BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played

class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    board = np.zeros((6,7), dtype=BoardPiece)
    return board

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    print('==============')
    for i in range(6):
        for j in range(7):
            if np.flipud(board)[i][j] == 0:
                print('.', end=" ")
            elif np.flipud(board)[i][j] == 1:
                print('X', end=" ")
            else:
                print('O', end=" ")

        print()
    print('==============', '\n0 1 2 3 4 5 6')
    #return np.str(board)

def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    return np.array(print_board(board))

def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    i = 0
    while board[i, action] != 0:
        i += 1

    if copy:
        board.copy()[i, action] = player
        return board.copy()
    else:
        board[i, action] = player
        return board

@njit()
def connected_four(
    board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    CONNECT_N = 4
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j:j+CONNECT_N] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i:i+CONNECT_N, j] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i+CONNECT_N, j:j+CONNECT_N]
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True
    return False


def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction]=None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    is_con = connected_four(board, player, last_action)
    if is_con:
        return GameState.IS_WIN
    elif np.sum(np.where(board==0)) == 0:
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING



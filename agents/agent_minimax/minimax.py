import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, GameState, connected_four, PLAYER1, PLAYER2, apply_player_action
import math
from typing import Optional, Callable

def evaluate_window(window, player):
    score = 0
    opp_player = PLAYER2
    if player == PLAYER2:
        opp_player = PLAYER1

    if window.count(player) == 4:
        score += 100
    elif window.count(player) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(player) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opp_player) == 3 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, player):
    score = 0
    center_column = board[:,3]
    center_count = center_column.count(player)
    score += center_count * 3

    ## Score Horizontal
    for r in range(6):
        row_array = board[r, :]
        for c in range(4):
            window = row_array[c:c+4]
            score += evaluate_window(window, player)

    # Score Vertical
    for c in range(7):
        col_array = board[:, c]
        for r in range(3):
            window = col_array[r:r+4]
            score += evaluate_window(window, player)

    # Score posiive sloped diagonal
    for r in range(3):
        for c in range(4):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, player)

    for r in range(3):
        for c in range(4):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, player)

    return score

depth = 8
alpha = -math.inf
beta = math.inf
maximizingPlayer = True

def alpha_beta(
        board, _player: BoardPiece, saved_state: Optional[SavedState], args=(depth, alpha, beta, maximizingPlayer)
):
    # Choose a valid, non-full column randomly and return it as `action`
    if depth == 0 or GameState.IS_DRAW or GameState.IS_WIN:
        if depth == 0:
            score_position(board, player)
        else:
            valid_columns = np.where(board[-1, :] == 0)
            column = PlayerAction(np.random.choice(np.array(valid_columns).flatten(), 1))
        return column, saved_state
        #if GameState.IS_WIN:
         #   if connected_four(board, PLAYER1, action) == True:
          #      return 1e15
           # if connected_four(board, PLAYER2, action) == True:
            #    return -1e15

        #if GameState.IS_DRAW:
         #   return None, 0
        #else:  # Depth is zero

         #   return None, score_position(board, PLAYER1)

    if maximizingPlayer:
        value = -math.inf
        valid_columns = np.where(board[-1, :] == 0)
        column = np.random.choice(np.array(valid_columns).flatten(), 1)
        for col in valid_columns:
            board_copy = board.copy()
            apply_player_action(board_copy, PlayerAction(col), player)
            new_score = minimax(board_copy, PLAYER1, depth - 1, alpha, beta, False)
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, saved_state

    else:  # Minimizing player
        value = math.inf
        column = np.random.choice(np.array(valid_columns).flatten(), 1)
        for col in valid_locations:
            board_copy = board.copy()
            apply_player_action(board_copy, PlayerAction(col), player)
            new_score = minimax(board_copy, PLAYER1, depth - 1, alpha, beta, True)
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, saved_state

def generate_move_minimax(
        board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]
):
    column = alpha_beta(board, player, *arga):
    return column


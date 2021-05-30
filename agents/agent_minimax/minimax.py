import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, GameState, connected_four, PLAYER1, PLAYER2, apply_player_action, check_end_state
import math
from typing import Optional, Callable, Tuple


def evaluate_window(window: np.array, player: BoardPiece) -> np.float:
    # Remark: please write docstrings for your functions
    W = list(window)
    score = 0
    opp_player = PLAYER2 # Remark: refactor this to its own function
    if player == PLAYER2:
        opp_player = PLAYER1

    # REmark: you can only have one of the cases below, so why is there a second if statement? There's also
    #         no need to initialize the heuristic_value and then update it, since you will only modify it once (i.e. you
    #         could just as well define it in each if/elif part
    # Remark: include more cases for the opponent, especially for a win
    # Remark: also, usually the heuristic is symmetric, i.e. 3 pieces for player and 1 for opponent should be the same value
    #         with opposite sign as 3 pieces for opponent and 1 for player.
    if W.count(player) == 4:
        score += 100
    elif W.count(player) == 3 and W.count(0) == 1:
        score += 5
    elif W.count(player) == 2 and W.count(0) == 2:
        score += 2

    if W.count(opp_player) == 3 and W.count(0) == 1:
        score -= 4

    return score


def score_position(board: np.ndarray, player: BoardPiece) -> np.float:
    # Remark: this function is quite long, try refactoring it
    score = 0
    center_column = list(board[:,3])
    center_count = center_column.count(player)
    score += center_count * 3
    # Remark: to be consistent, you should also account for the opponent here

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


def alpha_beta(
        board: np.ndarray, player: BoardPiece, depth: np.int, alpha: np.float, beta: np.float, maximizingPlayer: bool
) -> Tuple[PlayerAction, np.float]:
    # Choose a valid, non-full column randomly and return it as `action`
    valid_columns = np.where(board[-1, :] == 0)[0]  # Remark: this should be its own function, reused elsewhere
    opp_player = PLAYER2 if player == PLAYER1 else PLAYER1  # Remark: should also be a function on its own
    game_state = check_end_state(board, opp_player if maximizingPlayer else player)
    # Remark: you don't need to check for wins of both players, only the player who last moved can have a win.
    if depth == 0 or game_state in (GameState.IS_DRAW, GameState.IS_WIN):
        if game_state == GameState.IS_WIN:
            if maximizingPlayer:
                return PlayerAction(-1), -1000000000000
            else:
                return PlayerAction(-1), 1000000000000

        elif game_state == GameState.IS_DRAW:
            return PlayerAction(np.random.choice(np.array(valid_columns).flatten(), 1)), 0
        else:   # depth = 0
            return PlayerAction(np.random.choice(np.array(valid_columns).flatten(), 1)), score_position(board, player)

    if maximizingPlayer:
        value = -math.inf
        column = np.random.choice(np.array(valid_columns).flatten(), 1)  # Remark: randomizing here is not necessary, since you pick a move among moves later anyway.
        for col in valid_columns:
            # board_copy = board.copy()
            new_board = apply_player_action(board, PlayerAction(col), player, True)
            new_score = alpha_beta(new_board, opp_player, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return PlayerAction(column), value

    else:  # Minimizing player
        value = math.inf
        column = np.random.choice(np.array(valid_columns).flatten(), 1)
        for col in valid_columns:
            # board_copy = board.copy()
            new_board = apply_player_action(board, PlayerAction(col), player, True)
            new_score = alpha_beta(new_board, opp_player, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return PlayerAction(column), value


def generate_move_minimax(
    board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, SavedState]:
    # Choose a valid, non-full column randomly and return it as `action`
    depth = 4
    alpha = -math.inf
    beta = math.inf
    maximizingPlayer = True

    action = alpha_beta(board, _player, depth, alpha, beta, maximizingPlayer)[0]

    return PlayerAction(action), saved_state

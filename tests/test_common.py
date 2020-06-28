import numpy as np
import pytest
from agents.common import BoardPiece, NO_PLAYER, GameState
import pexpect

def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_pretty_print_board():
    from agents.common import pretty_print_board
    board = np.zeros((6, 7), dtype=BoardPiece)
    ret = pretty_print_board(board)
    assert isinstance(ret, np.str)

def test_string_to_board():
    from agents.common import string_to_board, pretty_print_board
    board = np.zeros((6, 7), dtype=BoardPiece)
    ret = string_to_board(pretty_print_board(board))
    assert isinstance(ret, np.ndarray)

def test_apply_player_action():
    from agents.common import apply_player_action, PlayerAction

    board = np.zeros((6, 7), dtype=BoardPiece)
    action = PlayerAction(2)
    player = BoardPiece(2)
    copy = True
    ret = apply_player_action(board, action, player, copy)
    assert isinstance(ret, np.ndarray)

def test_connected_four():
    from agents.common import connected_four
    board = np.zeros((6, 7), dtype=BoardPiece)
    player = BoardPiece(2)
    ret = connected_four(board, player)
    assert isinstance(ret, bool)

def test_check_end_state():
    from agents.common import check_end_state
    board = np.zeros((6, 7), dtype=BoardPiece)
    player = BoardPiece(2)
    ret = check_end_state(board, player)
    assert isinstance(ret, GameState)
import numpy as np
import pytest
from agents.common import BoardPiece, NO_PLAYER

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
    from agents.common import string_to_board
    ret = string_to_board
    assert isinstance(ret, np.ndarray)

def test_apply_player_action():
    from agents.common import apply_player_action
    ret = apply_player_action(board, action, player)
    assert isinstance(ret, np.ndarray)
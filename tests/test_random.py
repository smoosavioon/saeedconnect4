import numpy as np
from agents.common import PlayerAction, SavedState


def test_generate_move():
    from agents.agent_random import generate_move
    board = np.zeros((6, 7), dtype=BoardPiece)
    player = BoardPiece(2)
    ret = generate_move(board, player)
    assert isinstance(ret, tuple)

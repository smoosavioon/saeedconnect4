import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState

def test_evaluate_window():
    from agents.agent_minimax import evaluate_window
    window = np.array([0,1.0,1])
    player = BoardPiece(2)
    ret = evaluate_window(window, player)
    assert isinstance(ret, np.int)

def test_alpha_beta():
    from agents.agent_minimax import alpha_beta
    board = np.zeros((6, 7), dtype=BoardPiece)
    player = BoardPiece(2)
    depth = 5
    alpha = -math.inf
    beta = math.inf
    maximizingPlayer = True
    ret = alpha_beta(board, player, depth, alpha, beta, maximizingPlayer)
    assert isinstance(ret, tuple())

def test_score_position():
    from agents.agent_minimax import score_position
    board = np.zeros((6, 7), dtype=BoardPiece)
    player = BoardPiece(2)
    ret = score_position(board, player)
    assert isinstance(ret, np.int)

def test_generate_move_minimax():
    from agents.agent_minimax import generate_move_minimax
    board = np.zeros((6, 7), dtype=BoardPiece)
    player = BoardPiece(2)
    ret = generate_move_minimax(board, player)
    assert isinstance(ret, tuple())

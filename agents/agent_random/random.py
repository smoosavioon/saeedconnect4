import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from typing import Optional, Callable

def generate_move_random(
    board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]=None
) -> tuple[PlayerAction, SavedState]:
    # Choose a valid, non-full column randomly and return it as `action`

    valid_columns = np.where(board[-1,:] == 0)
    action = PlayerAction(np.random.choice(np.array(valid_columns).flatten(), 1))

    return action, saved_state
from chinese_checkers.env.game import ChineseCheckers, Move, Position
import numpy as np


def action_to_move(action: int, n: int):
    # index is flattened q, r, direction, is_jump
    # dim 4n+1 x 4n+1 x 6 x 2
    index = action
    index, is_jump = divmod(index, 2)
    index, direction = divmod(index, 6)
    _q, _r = divmod(index, 4 * n + 1)
    q, r = _q - 2 * n, _r - 2 * n
    return Move(q, r, direction, bool(is_jump))

def move_to_action(move: Move, n: int):
    # index is flattened q, r, direction, is_jump
    # dim 4n+1 x 4n+1 x 6 x 2
    q, r, direction, is_jump = move.position.q, move.position.r, move.direction, move.is_jump
    index = int(is_jump) + 2 * (direction + 6 * ((r + 2 * n) + (4 * n + 1) * (q + 2 * n)))
    return index

def get_legal_move_mask(board: ChineseCheckers, player: int):
    mask = np.zeros((4 * board.n + 1, 4 * board.n + 1, 6, 2), dtype=np.int8).flatten()
    for move in board.get_legal_moves(player):
        mask[move_to_action(move, board.n)] = 1
    return mask
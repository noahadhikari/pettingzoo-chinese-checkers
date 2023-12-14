import functools
import PIL

import numpy as np
from matplotlib import pyplot as plt
from enum import IntEnum

import gymnasium
from gymnasium.spaces import Box, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import pygame

from typing import Tuple

class Direction(IntEnum):
    Right = 0
    UpRight = 1
    UpLeft = 2
    Left = 3
    DownLeft = 4
    DownRight = 5

class Position:
    direction_map = {
        Direction.Right    : (+1,  0, -1),
        Direction.UpRight  : (+1, -1,  0),
        Direction.UpLeft   : ( 0, -1, +1),
        Direction.Left     : (-1,  0, +1),
        Direction.DownLeft : (-1, +1,  0),
        Direction.DownRight: ( 0, +1, -1)
    }

    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r
        self.s = -q - r

    def neighbor(self, direction: Direction, multiplier: int = 1):
        q_delta, r_delta, _ = Position.direction_map[direction]
        return Position(
            self.q + q_delta * multiplier,
            self.r + r_delta * multiplier
        )
    
    def __eq__(self, other):
        return self.q == other.q and self.r == other.r

class Move:
    END_TURN = "END_TURN"
    
    def __init__(self, q: int, r: int, direction: Direction, is_jump: bool):
        self.position = Position(q, r)
        self.direction = direction
        self.is_jump = is_jump
        
    def moved_position(self):
        multiplier = 2 if self.is_jump else 1
        return self.position.neighbor(self.direction, multiplier)
    
    @staticmethod
    def rotate60(move, times):
        assert move is not None
        q, r = move.position.q, move.position.r
        s = -q - r
        absolute_q, absolute_r, absolute_s = ChineseCheckers._rotate_60(q, r, s, -times)
        absolute_direction = (move.direction - times) % 6
        return Move(absolute_q, absolute_r, absolute_direction, move.is_jump)
    
    @staticmethod
    def to_absolute_move(move, player):
        return Move.rotate60(move, -player) if move != Move.END_TURN else Move.END_TURN
    
    @staticmethod
    def to_relative_move(move, player):
        return Move.rotate60(move, player) if move != Move.END_TURN else Move.END_TURN

    def __str__(self):
        if (self == Move.END_TURN):
            return "Move.END_TURN"
        return f"Move({self.position.q}, {self.position.r}, {self.direction}, {self.is_jump})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Move):
            return self.position == other.position and \
                   self.direction == other.direction and \
                   self.is_jump == other.is_jump
        else:
            return False
            

class ChineseCheckers:
    colors = {
        -1: (154, 132, 73), # empty cell
        0: (255, 0, 0), # red
        1: (0, 0, 0), # black
        2: (255, 255, 0), # yellow
        3: (0, 255, 0), # green
        4: (0, 0, 255), # blue
        5: (255, 255, 255), # white
    }

    OUT_OF_BOUNDS = -2
    EMPTY_SPACE = -1

    def __init__(self, triangle_size: int, render_mode: str = "rgb_array"):
        r"""
        At the beginning of the game:
            Player 0 is at the top of the board,
            Player 1 is top-right,
            Player 2 is bottom-right,
            Player 3 is bottom,
            Player 4 is bottom-left,
            Player 5 is top-left:

               0
            5     1
            4     2
               3

            (q, r, s) = (0, 0, 0) in cube coordinates is the center of the board.
            +q-axis is top-right (/), +r-axis is straight down (|), +s-axis is towards top-left (\).
            These correspond to the LINES of the hexagons, not the centers.

            +s -r +q
              \ | /
                o
              / | \
            -q +r -s
            

            Cube coordinates satisfy the constraint x + y + z = 0.
            
            Each player's pegs are in a triangle with side length 4. The center of the board is an unoccupied hexagon of side length 5.

            We number each peg starting from the top of the triangle in level-order, for example, Player 1's pegs are
            as follows:
            
                  9
                5   8
              2   4   7
            0   1   3   6

            This corresponds to the following relative cube coordinates, with bottom-left at (0, 0, 0):
                0: (0, 0, 0)
                1: (1, -1, 0)
                2: (1, 0, -1)
                3: (2, -2, 0)
                4: (2, -1, -1)
                5: (2, 0, -2)
                6: (3, -3, 0)
                7: (3, -2, -1)
                8: (3, -1, -2)
                9: (3, 0, -3)
            
            The other initial positions are similar but rotated clockwise about the origin by 60 degrees for each player.

        """
        self.render_mode = render_mode
        self.n = triangle_size
        self.rotation = 0
        self.clock = None

        self.init_game()

        # Players 0 through 5 are the six players
        self.window_size = 512  # The size of the PyGame window
        self.window = None

    def _set_rotation(self, player: int):
        # Set's board rotation so that the kth player is at the top of the board
        self.rotation = player
        self._jumps = list(map(lambda j: Move.to_relative_move(j, player), self._jumps))

    def _unset_rotation(self):
        assert self.rotation is not None
        self._jumps = list(map(lambda j: Move.to_relative_move(j, -self.rotation), self._jumps))
        self.rotation = None

    @staticmethod
    def _rotate_60(q: int, r: int, s: int, times: int):
        """Rotates clockwise."""
        for _ in range(times % 6):
            q, r, s = -r, -s, -q
        return q, r, s

    @staticmethod
    def _rotate_60_about_pt(q: int, r: int, s: int, o_q: int, o_r: int, o_s: int, times: int):
        """Rotates clockwise."""
        n_q = q - o_q
        n_r = r - o_r
        n_s = s - o_s
        n_q, n_r, n_s = ChineseCheckers._rotate_60(n_q, n_r, n_s, times)
        return n_q + o_q, n_r + o_r, n_s + o_s

    def init_game(self):
        """
        Board is a 3D array of shape (4n + 1, 4n + 1, 4n + 1), with each element being a number from -2 to 5, inclusive.
             -2: invalid space
             -1: empty space
            0-5: player number of occupying peg
        """
        self._jumps = []
        self._legal_moves = None
        self._game_over = False

        def _fill_center_empty():
            self._set_rotation(0)
            for q in range(-self.n, self.n + 1):
                for r in range(-self.n, self.n + 1):
                    s = -q - r
                    if abs(q) + abs(r) + abs(s) <= 2 * self.n:
                        self._set_coordinate(q, r, s, ChineseCheckers.EMPTY_SPACE)
            self._unset_rotation()

        def _fill_home_triangle(player: int):
            self._set_rotation(player)
            for q, r, s in self._get_home_coordinates():
                self._set_coordinate(q, r, s, player)
            self._unset_rotation()
        
        # Set the whole board to invalid
        self.board = ChineseCheckers.OUT_OF_BOUNDS * np.ones((4 * self.n + 1, 4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)

        # Fill player starting triangles
        for player in range(6):
            _fill_home_triangle(player)

        # Fill center with empty spaces
        _fill_center_empty()
            
    def find_legal_moves(self, player: int):
        """
        Changes the game's legal moves to a list of legal moves for the specified player.
        """
        self._set_rotation(player)
        moves = []
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                for direction in Direction:
                    for is_jump in [False, True]:
                        move = Move(q, r, direction, is_jump)
                        if self._is_single_move_legal(move):
                            moves.append(move)
        
        if (len(moves) == 0 and not self._game_over) or self._is_single_move_legal(Move.END_TURN):
            moves.append(Move.END_TURN)
            
        self._legal_moves = moves
        self._unset_rotation()
    
    def get_legal_moves(self, player: int):
        if self._legal_moves is None:
            self.find_legal_moves(player)
        return self._legal_moves

    def _get_home_coordinates(self):
        """
        Returns (q, r, s) tuples for the absolute coordinates of the player's home triangle.
        Has relative coordinate (0, 0, 0) as the leftmost point of the triangle for player 0.
        """
        assert self.rotation is not None
        offset = np.array([1, -self.n - 1, self.n])
        for i in range(self.n):
            for j in range(i, self.n):
                q, r, s = j, -i, i - j
                yield offset + np.array([q, r, s])
    
    def _home_values(self):
        """
        Generates all values within the player's target triangle.
        """
        assert self.rotation is not None
        for q, r, s in self._get_home_coordinates():
            yield self._get_board_value(q, r, s)

    def _get_target_coordinates(self):
        """
        Returns (q, r, s) tuples for the absolute coordinates of the current player's target triangle.
        Has relative coordinate (0, 0, 0) as the leftmost point of the triangle for player 0.
        """
        assert self.rotation is not None
        offset = np.array([-self.n, self.n + 1, -1])
        for i in range(self.n):
            for j in range(0, self.n - i):
                q, r, s = j, i, -i - j
                yield offset + np.array([q, r, s])

    def get_target_coordinates(self, player):
        """
        Returns (q, r, s) tuples for the absolute coordinates of the player's target triangle.
        Has relative coordinate (0, 0, 0) as the leftmost point of the triangle for player 0.
        """
        self._set_rotation(player)
        yield from self._get_target_coordinates()
        self._unset_rotation()

    def _target_values(self):
        """
        Generates all values within the player's target triangle.
        """
        assert self.rotation is not None
        for q, r, s in self._get_target_coordinates():
            yield self._get_board_value(q, r, s)
    
    def _in_bounds(self, q: int, r: int, s: int):
        board_q, board_r, board_s = q + 2 * self.n, r + 2 * self.n, s + 2 * self.n
        if (board_q < 0 or board_q >= 4 * self.n + 1 or \
            board_r < 0 or board_r >= 4 * self.n + 1 or \
            board_s < 0 or board_s >= 4 * self.n + 1):
            return False
        
        start = self._get_board_value(q, r, s)
        return start != ChineseCheckers.OUT_OF_BOUNDS
        
    def _get_board_value(self, q: int, r: int, s: int):
        """
        Given q, r, s coordinates in the current viewing perspective, get the board value.
        """
        assert self.rotation is not None
        rotated_q, rotated_r, rotated_s = self._rotate_60(q, r, s, self.rotation)
        board_q, board_r, board_s = rotated_q + 2 * self.n, rotated_r + 2 * self.n, rotated_s + 2 * self.n
        return self.board[board_q, board_r, board_s]

    def _set_coordinate(self, q: int, r: int, s: int, value: int):
        """
        Given q, r, s coordinates in the current viewing perspective, set the board value.
        """
        assert self.rotation is not None
        rotated_q, rotated_r, rotated_s = self._rotate_60(q, r, s, self.rotation)
        board_q, board_r, board_s = rotated_q + 2 * self.n, rotated_r + 2 * self.n, rotated_s + 2 * self.n
        self.board[board_q, board_r, board_s] = value
    
    @staticmethod
    def _cube_to_axial(q: int, r: int, s: int):
        return q, r
    
    @staticmethod
    def _axial_to_cube(q: int, r: int):
        return q, r, -q - r
    
    def _get_axial_board(self):
        r""""Helper function for get_axial_board"""
        assert self.rotation is not None 
        result = -2 * np.ones((4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                s = -q - r
                if abs(q) + abs(r) + abs(s) <= 4 * self.n + 1:
                    result[q, r] = self._get_board_value(q, r, s)
        result = np.vectorize(self._rotate_player_number)(result)
        return result   

    def _is_player(self, value):
        return 0 <= value < 6

    def _rotate_player_number(self, peg):
        assert self.rotation is not None
        return (peg - self.rotation) % 6 if self._is_player(peg) else peg
    
    def get_axial_board(self, player):
        r"""
        Returns the current game board in axial coordinates from the perspective of the specified player.
        """
        self._set_rotation(player)
        board = self._get_axial_board()
        self._unset_rotation()
        return board
    
    def _is_single_move_legal(self, move: Move):
        """
        Helper function for `is_move_legal` that assumes the player's home triangle
        is at the top of the board.
        Checks if a move is legal. Does not consider the edge case in which no moves are valid.
        """
        assert self.rotation is not None

        if self._game_over:
            return False

        player = self.rotation

        # END_TURN is only legal if the last move was a jump
        if (move == Move.END_TURN):
            return len(self._jumps) > 0

        # Check that the start position is in bounds for the star board
        if (not self._in_bounds(move.position.q, move.position.r, move.position.s)):
            return False
        
        # The start position must contain one of the player's pegs
        start_position_value = self._get_board_value(
            move.position.q, 
            move.position.r, 
            move.position.s
        )
        if (start_position_value != player):
            return False
        
        # The moved to position must be in bounds
        moved_to = move.moved_position()
        if (not self._in_bounds(moved_to.q, moved_to.r, moved_to.s)):
            return False
        
        # The moved to value must be an empty space
        moved_to_value = self._get_board_value(moved_to.q, moved_to.r, moved_to.s)
        if (moved_to_value != ChineseCheckers.EMPTY_SPACE):
            return False
        
        # If an move is a jump, there must be a direct neighbor between the
        # start position and the moved to position.
        if move.is_jump:
            direct_neighbor = move.position.neighbor(move.direction)
            if (not self._in_bounds(direct_neighbor.q, direct_neighbor.r, direct_neighbor.s)):
                return False
            direct_neighbor_value = self._get_board_value(
                direct_neighbor.q, 
                direct_neighbor.r, 
                direct_neighbor.s
            )
            if (direct_neighbor_value == ChineseCheckers.EMPTY_SPACE):
                return False
            
            if (len(self._jumps) > 0):
                # Can only move one peg per turn
                last_jump = self._jumps[-1]
                prev_jumped_position = last_jump.position.neighbor(last_jump.direction, 2)
                if (prev_jumped_position != move.position):
                    return False
                
                # Prevent jumping infinitely in cycles
                jumped_position = move.position.neighbor(move.direction, 2)
                for prev_jump in self._jumps:
                    if (jumped_position == prev_jump.position):
                        return False
        else:
            # If the move is not a jump, we can't have jumped any other time this turn
            if len(self._jumps) != 0:
                return False

        # Passed all checks
        return True
    

    # Checks whether a move is legal for the current player
    def is_move_legal(self, move: Move, player: int) -> bool:
        if self._legal_moves is None:
            self.find_legal_moves(player)
        self._set_rotation(player)
        is_legal = move in self._legal_moves
        self._unset_rotation()
        return is_legal
    
    # Executes the move for the specified player. Errors if move is illegal.
    # Returns the number of the player's pegs within the target triangle.
    def move(self, player: int, move: Move) -> int:
        if (not self.is_move_legal(move, player)):
            print(f"{move} is not legal for player {player}")
            return None
        # assert self.is_move_legal(move, player), f"{move} is not legal for player {player}"
        if move == Move.END_TURN:
            self._jumps.clear()
            self._legal_moves = None
            return move
        
        self._set_rotation(player)
        # score = 0
        src_pos = move.position
        self._set_coordinate(src_pos.q, src_pos.r, src_pos.s, ChineseCheckers.EMPTY_SPACE)
        dst_pos = move.moved_position()

        self._set_coordinate(dst_pos.q, dst_pos.r, dst_pos.s, player)
        if move.is_jump:
            self._jumps.append(move)
        else:
            self._jumps.clear()
        self._legal_moves = None
        if self._did_player_win():
            self._game_over = True
        self._unset_rotation()
        return move

    def get_jumps(self, player: int):
        return [Move.to_relative_move(jump, player) for jump in self._jumps]

    def get_last_jump(self, player: int):
        """
        Gets the last jump in the specified player's frame if it exists, otherwise None
        """
        if len(self._jumps) > 0:
            return Move.to_relative_move(self._jumps[-1], player)

    def _did_player_win(self) -> bool:
        return all([value == self.rotation for value in self._target_values()])

    def did_player_win(self, player: int) -> bool:
        self._set_rotation(player)
        did_win = self._did_player_win()
        self._unset_rotation()
        return did_win
    
    def is_game_over(self) -> bool:
        return self._game_over
    
    def render(self, player: int = 0):
        self._set_rotation(player)
        frame = None
        frame = self._render_frame()
        self._unset_rotation()
        return frame

    def _render_frame(self):
        """
        Renders a frame of the game. https://www.gymlibrary.dev/content/environment_creation/#rendering
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((241, 212, 133)) # Fill background

        def axial_to_pixel(q: int, r: int):
            l = 20
            screen_center_x, screen_center_y = self.window_size / 2, self.window_size / 2
            return screen_center_x + l * np.sqrt(3) * (q + 0.5 * r), \
                screen_center_y + l * 1.5 * r

        axial_board = self._get_axial_board()
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                pixel_x, pixel_y = axial_to_pixel(q, r)
                cell = axial_board[q, r]
                if cell == ChineseCheckers.OUT_OF_BOUNDS:
                    # Not a valid cell
                    continue
                else:
                    # Cell with a peg
                    pygame.draw.circle(
                        canvas,
                        self.colors[cell],
                        (pixel_x, pixel_y),
                        8,
                    )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(12)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for i, size in enumerate([2, 3, 4]):
        game = ChineseCheckers(size, "rgba_array")
        frame = game.render()
        ax[i].imshow(frame)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(rf"$N$ = {size}")
    plt.tight_layout()
    plt.savefig("frame.png")
    
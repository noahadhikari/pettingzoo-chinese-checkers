import functools

import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from gymnasium.spaces import Box, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import pygame

class ChineseCheckersPosition:
    direction_map = {
        0: (+1,  0, -1),
        1: (+1, -1,  0),
        2: ( 0, -1, +1),
        3: (-1,  0, +1),
        4: (-1, +1,  0),
        5: ( 0, +1, -1)
    }

    def __init__(self, q, r):
        self.q = q
        self.r = r
        self.s = -q - r

    def neighbor(self, direction, multiplier=1):
        direction_delta = ChineseCheckersPosition.direction_map[direction]
        return ChineseCheckersPosition(
            self.q + direction_delta[0] * multiplier,
            self.r + direction_delta[1] * multiplier
        )

class ChineseCheckersMove:
    def __init__(self, q, r, direction, is_jump):
        self.position = ChineseCheckersPosition(q, r)
        self.direction = direction
        self.is_jump = is_jump
        
    def moved_position(self):
        multiplier = 2 if self.is_jump else 1
        return self.position.neighbor(self.direction, multiplier)

class ChineseCheckers:
    colors = {
        -1: (154, 132, 73),
        0: (255, 0, 0),
        1: (0, 0, 0),
        2: (255, 255, 0),
        3: (0, 255, 0),
        4: (0, 0, 255),
        5: (255, 255, 255),
    }

    OUT_OF_BOUNDS = -2
    EMPTY_SPACE = -1

    def __init__(self, triangle_size: int):

        # Players 0 through 5 are the six players
        self.window_size = 512  # The size of the PyGame window
        self.window = None
        """
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
        self.n = triangle_size
        self.rotation = 0
        self.init_board()

    def _set_rotation(self, player: int):
        # Set's board rotation so that the kth player is at the top of the board
        self.rotation = player

    @staticmethod
    def _rotate_60(q: int, r: int, s: int, times: int):
        """Rotates clockwise."""
        for _ in range(times):
            q, r, s = -r, -s, -q
        return q, r, s

    @staticmethod
    def _rotate_60_about_pt(q: int, r: int, s: int, o_q: int, o_r: int, o_s: int, times: int):
        """Rotates clockwise."""
        n_q = q - o_q
        n_r = r - o_r
        n_s = s - o_s
        n_q, n_r, n_s = raw_env._rotate_60(n_q, n_r, n_s, times)
        return n_q + o_q, n_r + o_r, n_s + o_s

    def init_board(self):
        """
        Board is a 3D array of shape (4n + 1, 4n + 1, 4n + 1), with each element being a number from -2 to 5, inclusive.

             -2: invalid space
             -1: empty space
            0-5: player number of occupying peg
        """
        # Set the whole board to invalid
        self.board = ChineseCheckers.OUT_OF_BOUNDS * np.ones((4 * self.n + 1, 4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)

        # Fill player starting triangles
        for player in range(6):
            self._set_rotation(player)
            self._fill_home_triangle(player)
        self._set_rotation(0)

        # Fill center with empty spaces
        self._fill_center_empty()
        
        self.game_over = False
        
    def _fill_center_empty(self):
        for q in range(-self.n, self.n + 1):
            for r in range(-self.n, self.n + 1):
                s = -q - r
                if abs(q) + abs(r) + abs(s) <= 2 * self.n:
                    self._set_coordinate(q, r, s, ChineseCheckers.EMPTY_SPACE)

    def _fill_home_triangle(self, player: int):
        for q, r, s in self._get_home_coordinates():
            self._set_coordinate(q, r, s, player)
    
    def _get_home_coordinates(self):
        """
        Returns (q, r, s) tuples for the absolute coordinates of the player's home triangle.
        Has relative coordinate (0, 0, 0) as the leftmost point of the triangle for player 0.
        """
        result = []
        offset = np.array([1, -self.n - 1, self.n])
        for i in range(self.n):
            for j in range(i, self.n):
                q, r, s = j, -i, i - j
                result.append(offset + np.array([q, r, s]))

        return result
    
    def _get_target_coordinates(self):
        """
        Returns (q, r, s) tuples for the absolute coordinates of the player's home triangle.
        Has relative coordinate (0, 0, 0) as the leftmost point of the triangle for player 0.
        """
        result = []
        offset = np.array([-self.n, self.n + 1, -1])
        for i in range(self.n):
            for j in range(0, self.n - i):
                q, r, s = j, i, -i - j
                result.append(offset + np.array([q, r, s]))

        return result
    
    def _in_bounds(self, q: int, r: int, s: int):
        rotated_q, rotated_r, rotated_s = self._rotate_60(q, r, s, self.rotation)
        board_q, board_r, board_s = rotated_q + 2 * self.n, rotated_r + 2 * self.n, rotated_s + 2 * self.n
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
        rotated_q, rotated_r, rotated_s = self._rotate_60(q, r, s, self.rotation)
        board_q, board_r, board_s = rotated_q + 2 * self.n, rotated_r + 2 * self.n, rotated_s + 2 * self.n
        return self.board[board_q, board_r, board_s]

    def _set_coordinate(self, q: int, r: int, s: int, value: int):
        """
        Given q, r, s coordinates in the current viewing perspective, set the board value.
        """
        rotated_q, rotated_r, rotated_s = self._rotate_60(q, r, s, self.rotation)
        board_q, board_r, board_s = rotated_q + 2 * self.n, rotated_r + 2 * self.n, rotated_s + 2 * self.n
        self.board[board_q, board_r, board_s] = value

    def render(self, player: int = 0):
        self._set_rotation(player)
        frame = self._render_frame()
        self._set_rotation(0)
        return frame

    def _render_frame(self):
        """
        Renders a frame of the game. https://www.gymlibrary.dev/content/environment_creation/#rendering
        """
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((241, 212, 133)) # Fill background

        def cubic_to_pixel(q: int, r: int, s: int):
            l = 20
            screen_center_x, screen_center_y = self.window_size / 2, self.window_size / 2
            return screen_center_x + l * np.sqrt(3) * (q + 0.5 * r), \
                screen_center_y + l * 1.5 * r

        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                for s in range(-2 * self.n, 2 * self.n + 1):
                    pixel_x, pixel_y = cubic_to_pixel(q, r, s)
                    cell = self._get_board_value(q, r, s)
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

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    @staticmethod
    def _cube_to_axial(q: int, r: int, s: int):
        return q, r
    
    @staticmethod
    def _axial_to_cube(q: int, r: int):
        return q, r, -q - r
    
    def _axial_board(self):
        result = -2 * np.ones((4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                s = -q - r
                if abs(q) + abs(r) + abs(s) <= 2 * self.n:
                    result[q, r] = self._get_board_value(q, r, s)
        return result
    
    def axial_board(self, player):
        self._set_rotation(player)
        board = self._axial_board()
        self._set_rotation(0)
        return board
    
    # Helper function for `is_move_legal` that assumes the player's home triangle
    # is at the top of the board.
    def _is_move_legal(self, player, move: ChineseCheckersMove):
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
            print(start_position_value, move.position.q, move.position.r, player)
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
        return True
    
    # Checks whether a move is legal for the specified player
    def is_move_legal(self, player: int, move: ChineseCheckersMove) -> bool:
        self._set_rotation(player)
        is_legal = self._is_move_legal(player, move)
        self._set_rotation(0)
        return is_legal
    
    # Executes the move for the specified player. Errors if move is illegal.
    # Returns the number of the player's pegs within the target triangle.
    def move(self, player: int, move: ChineseCheckersMove) -> int:
        assert self.is_move_legal(player, move)
        score = 0
        self._set_rotation(player)
        src_pos = move.position
        self._set_coordinate(src_pos.q, src_pos.r, src_pos.s, ChineseCheckers.EMPTY_SPACE)
        dst_pos = move.moved_position()
        for q, r, s in self._get_target_coordinates():
            if self._get_board_value(q, r, s) == player:
                score += 1
        self._set_coordinate(dst_pos.q, dst_pos.r, dst_pos.s, player)
        self._set_rotation(0)
        return score

    def did_player_win(self, player: int) -> bool:
        self._set_rotation(player)
        did_win = True
        for q, r, s in self._get_target_coordinates(q, r, s):
            if self._get_board_value(q, r, s) != player:
                did_win = False
                break
        self._set_rotation(0)
        if did_win:
            self.game_over = True
        return did_win
    
    def is_game_over(self) -> bool:
        return self.game_over
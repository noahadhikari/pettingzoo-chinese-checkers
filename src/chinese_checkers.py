import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

def env(triangle_size=4):
    env = raw_env(triangle_size)
    return env

class raw_env(AECEnv):

    metadata = {"render_modes": ["human"], "name": "chinese_checkers"}

    def __init__(self, triangle_size):

        # Players 0 through 5 are the six players
        self.agents = [r for r in range(6)]

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
        self._init_board()

    @staticmethod
    def _rotate_60(q, r, s, times):
        """Rotates clockwise."""
        for _ in range(times):
            q, r, s = -r, -s, -q
        return q, r, s

    @staticmethod
    def _rotate_60_about_pt(q, r, s, o_q, o_r, o_s, times):
        """Rotates clockwise."""
        n_q = q - o_q
        n_r = r - o_r
        n_s = s - o_s
        raw_env._rotate_60(n_q, n_r, n_s, times)
        return n_q + o_q, n_r + o_r, n_s + o_s

    def _init_board(self):
        """
        Board is a 3D array of shape (4n + 1, 4n + 1, 4n + 1), with each element being a number from -2 to 5, inclusive.

             -2: invalid space
             -1: empty space
            0-5: player number of occupying peg
        """
        # Set the whole board to invalid
        self.board = -2 * np.ones((4 * self.n + 1, 4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)

        # Fill player starting triangles
        for player in range(6):
            self._fill_triangle(player)

        # Fill center with empty spaces
        self._fill_center_empty()
        
    def _fill_center_empty(self):
        for q in range(-self.n, self.n + 1):
            for r in range(-self.n, self.n + 1):
                s = -q - r
                if abs(q) + abs(r) + abs(s) <= 2 * self.n:
                    self._set_coordinate(q, r, s, -1)

    def _fill_triangle(self, player):
        for x, y, z in self._get_home_coordinates(player):
            self._set_coordinate(x, y, z, player)
    
    def _get_home_coordinates(self, player):
        """
        Returns (x,y,z) tuples for the absolute coordinates of the player's home triangle.
        Has relative coordinate (0, 0, 0) as the leftmost point of the triangle for player 0.

        Then rotates the triangle CW by 60 degrees for each player.
        """
        result = []
        player_0_offset = np.array([1, -self.n - 1, self.n])
        for i in range(self.n):
            for j in range(0, self.n - i):
                q, r, s = j, -i, i - j
                result.append(self._rotate_60(*(player_0_offset + np.array([q, r, s])), player))

        return result

    def _get_coordinate(self, x, y, z):
        board_x, board_y, board_z = x + 2 * self.n, y + 2 * self.n, z + 2 * self.n
        return self.board[board_x][board_y][board_z]

    def _set_coordinate(self, x, y, z, value):
        board_x, board_y, board_z = x + 2 * self.n, y + 2 * self.n, z + 2 * self.n
        self.board[board_x][board_y][board_z] = value

    def reset(self, seed=None, options=None):
        self.init_board()

    def step(self, actions):

        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observe(agent)
    
    @staticmethod
    def _cube_to_axial(q, r, s):
        return q, r
    
    @staticmethod
    def _axial_to_cube(q, r):
        return q, r, -q - r
    
    def _compute_axial_board(self):
        result = -2 * np.ones((4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                s = -q - r
                result[q, r] = self._get_coordinate(q, r, s)
        return result

    def observe(self, agent):
        """ The observation is just the board, regardless of agent.
            To counteract the sparsity of cube coordinates, we convert it to axial
                (q, r, s) -> (q, r)
            first.
        """
        return self._compute_axial_board()

    def action_space(self, agent):
        # (4 * n + 1)^2 spaces in the (axial) board, 6 directions to move for each, 2 types (no-jump/jump)
        return Discrete(2 * 6 * (4 * self.n + 1) * (4 * self.n + 1))

if __name__ == "__main__":
    env = env(2)
    board = env.observe(0)
    for i, a in enumerate(board):
        for j, b in enumerate(a):
            for k, c in enumerate(b):
                if c != -2:
                    print(i - env.n - 1, j - env.n - 1, k - env.n - 1, c)
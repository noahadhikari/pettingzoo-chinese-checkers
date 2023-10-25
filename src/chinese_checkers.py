import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

def env():
    env = raw_env()
    return env

class raw_env(AECEnv):

    metadata = {"render_modes": ["human"], "name": "chinese_checkers"}

    def __init__(self):

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
            
            The other players' initial positions are similar.

        """
        self.n = 4
        self._init_board()

    def _init_board(self):
        """
        Board is a 3D array of shape (4n + 1, 4n + 1, 4n + 1), with each element being a number from -2 to 5, inclusive.

             -2: invalid space
             -1: empty space
            0-5: player number of occupying peg

        The observation space is just this board.
        """
        # Set the whole board to invalid
        self.board = -2 * np.ones((4 * self.n + 1, 4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)

        # Fill the player starting triangles
        for player in range(6):
            self._fill_triangle(player)

        # Fill the center with empty spaces
        self._fill_center_empty()
        
    def _fill_center_empty(self):
        for q in range(-self.n, self.n + 1):
            for r in range(-self.n, self.n + 1):
                s = -q - r
                if abs(q) + abs(r) + abs(s) <= 2 * self.n:
                    self._set_coordinate(q, r, s, -1)


    def _get_home_offset(self, player):
        if player == 0:
            return (1, -self.n - 1, self.n)
        elif player == 1:
            return (self.n + 1, -self.n, -1)
        elif player == 2:
            return (1, self.n, -self.n - 1)
        elif player == 3:
            return (-self.n, self.n + 1, -1)
        elif player == 4:
            return (-2 * self.n, self.n, self.n)
        elif player == 5:
            return (-self.n, -self.n, 2 * self.n)

    def _fill_triangle(self, player):
        for coords in self._get_home_coordinates(player):
            offset = self._get_home_offset(player)
            self._set_coordinate(**(offset + coords), player)

    def _get_home_coordinates(self, player):
        """
        Returns (x,y,z) tuples for the relative coordinates of the player's home triangle.
        Has (0, 0, 0) as the leftmost point of the triangle. Upward-facing for even players,
        downward-facing for odd players.
        """
        result = []

        if player % 2 == 0:
            for i in range(self.n):
                for j in range(0, self.n - i):
                    q, r, s = j, -i, i - j
                    result.append((q, r, s))
        else:
            for i in range(self.n):
                for j in range(0, self.n - i):
                    q, r, s = j, i, -i - j
                    result.append((q, r, s))

        return result

        
    def _triangle_number(n):
        return n * (n + 1) / 2

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
    
    def observe(self, agent):
        return self.obs

    def _get_actions(self, q, r, s):
        pass

    def action_space(self, agent):
        # (4 * n + 1)^3 spaces in the board, 6 directions to move for each, 2 types (no-jump/jump)
        return Discrete(2 * 6 * (4 * self.n + 1) * (4 * self.n + 1) * (4 * self.n + 1))

if __name__ == "__main__":
    result = []

    for y in range(4):
        for z in range(y + 1):
            result.append((y - z, y, z))
        
    # for y in range(self.n):
    #     for z in range(y + 1):
    #         result.append((z - y, y, -z))

    print(result)
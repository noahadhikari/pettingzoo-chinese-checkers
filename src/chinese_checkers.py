import functools

import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

import pygame

def env(triangle_size=4, render_mode="rgb_array"):
    env = raw_env(triangle_size, render_mode)
    return env

class raw_env(AECEnv):

    metadata = {"render_modes": ["rgb_array"], "name": "chinese_checkers"}
    colors = {
        0: (255, 0, 0),
        1: (0, 0, 0),
        2: (255, 255, 0),
        3: (0, 255, 0),
        4: (0, 0, 255),
        5: (255, 255, 255),
    }

    def __init__(self, triangle_size, render_mode):

        # Players 0 through 5 are the six players
        self.possible_agents = [f"player_{r}" for r in range(6)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.render_mode = render_mode
        self.window_size = 512  # The size of the PyGame window


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        https://www.gymlibrary.dev/content/environment_creation/#rendering
        """
        self.window = None
        self.clock = None

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
        self._init_board()

    def _set_rotation(self, k):
        # Set's board rotation so that the kth player is at the top of the board
        self.rotation = k

    @staticmethod
    def _rotate_60(q, r, s, times):
        """Rotates clockwise."""
        for _ in range(times):
            q, r, s = -r, -s, -q
        return q, r, s

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

        # Fill player starting triangles
        for player in range(6):
            self._set_rotation(player)
            self._fill_home_triangle(player)

        # Fill center with empty spaces
        self._fill_center_empty()
        
    def _fill_center_empty(self):
        for q in range(-self.n, self.n + 1):
            for r in range(-self.n, self.n + 1):
                s = -q - r
                if abs(q) + abs(r) + abs(s) <= 2 * self.n:
                    self._set_coordinate(q, r, s, -1)

    def _fill_home_triangle(self, player):
        for x, y, z in self._get_home_coordinates():
            self._set_coordinate(x, y, z, player)
    
    def _get_home_coordinates(self):
        """
        Returns (x,y,z) tuples for the absolute coordinates of the player's home triangle.
        Has relative coordinate (0, 0, 0) as the leftmost point of the triangle for player 0.

        Then rotates the triangle CW by 60 degrees for each player.
        """
        result = []
        player_0_offset = np.array([1, -self.n - 1, self.n])
        for i in range(self.n):
            for j in range(i, self.n):
                q, r, s = j, -i, i - j
                result.append(player_0_offset + np.array([q, r, s]))

        return result

    def _get_coordinate(self, x, y, z):
        """
        Given x, y, z coordinates in the current viewing perspective, get the board value.
        """
        rotated_x, rotated_y, rotated_z = self._rotate_60(x, y, z, self.rotation)
        board_x, board_y, board_z = rotated_x + 2 * self.n, rotated_y + 2 * self.n, rotated_z + 2 * self.n
        return self.board[board_x, board_y, board_z]

    def _set_coordinate(self, x, y, z, value):
        """
        Given x, y, z coordinates in the current viewing perspective, set the board value.
        """
        rotated_x, rotated_y, rotated_z = self._rotate_60(x, y, z, self.rotation)
        board_x, board_y, board_z = rotated_x + 2 * self.n, rotated_y + 2 * self.n, rotated_z + 2 * self.n
        self.board[board_x, board_y, board_z] = value

    def reset(self, seed=None, options=None):
        self.init_board()
        self.rotation = 0

    def step(self, actions):
        pass

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((241, 212, 133))

        def cubic_to_pixel(r, q, s):
            l = 20
            screen_center_x, screen_center_y = self.window_size / 2, self.window_size / 2
            return screen_center_x + l * np.sqrt(3) * (r + 0.5 * q), \
                screen_center_y + l * 1.5 * q

        for r in range(-2 * self.n, 2 * self.n + 1):
            for q in range(-2 * self.n, 2 * self.n + 1):
                for s in range(-2 * self.n, 2 * self.n + 1):
                    pixel_x, pixel_y = cubic_to_pixel(r, q, s)
                    cell = self._get_coordinate(r, q, s)
                    if cell == -2:
                        continue
                    elif cell == -1:
                        pygame.draw.circle(
                            canvas,
                            (154, 132, 73),
                            (pixel_x, pixel_y),
                            8,
                        )
                    else:
                        # Now we draw the agent
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
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def observation_space(self, agent):
        return self.observe(agent)
    
    def observe(self, agent):
        return self.board

    def action_space(self, agent):
        # (4 * n + 1)^3 spaces in the board, 6 directions to move for each, 2 types (no-jump/jump)
        return Discrete(2 * 6 * (4 * self.n + 1) * (4 * self.n + 1) * (4 * self.n + 1))
    
    def _print_board(self):
        for r in range(-2 * self.n, 2 * self.n + 1):
            for q in range(-self.n, 2 * self.n + 1):
                s = -r - q
                print(self._get_coordinate(r, q, s), end=" ")
            print()

if __name__ == "__main__":
    env = env(2)
    board = env.observe(0)
    # for i, a in enumerate(board):
    #     for j, b in enumerate(a):
    #         for k, c in enumerate(b):
    #             if c != -2:
    #                 print(i - env.n - 1, j - env.n - 1, k - env.n - 1, c)
    frame = env.render()
    plt.imshow(frame)
    plt.savefig("test.png")

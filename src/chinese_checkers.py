import functools

import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from gymnasium.spaces import Box, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import pygame

def env(triangle_size=4):
    env = raw_env(triangle_size)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

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

class ChineseCheckersAction:
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

    def _set_rotation(self, k: int):
        # Set's board rotation so that the kth player is at the top of the board
        self.rotation = k

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
        player_0_offset = np.array([1, -self.n - 1, self.n])
        for i in range(self.n):
            for j in range(i, self.n):
                q, r, s = j, -i, i - j
                result.append(player_0_offset + np.array([q, r, s]))

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
    
    def _is_action_legal(self, player, action: ChineseCheckersAction):
        # Check that the start position is in bounds for the star board
        if (not self._in_bounds(action.position.q, action.position.r, action.position.s)):
            return False
        
        # The start position must contain one of the player's pegs
        start_position_value = self._get_board_value(
            action.position.q, 
            action.position.r, 
            action.position.s
        )
        if (start_position_value != player):
            print(start_position_value, action.position.q, action.position.r, player)
            return False
        
        # The moved to position must be in bounds
        moved_to = action.moved_position()
        if (not self._in_bounds(moved_to.q, moved_to.r, moved_to.s)):
            return False
        
        # The moved to value must be an empty space
        moved_to_value = self._get_board_value(moved_to.q, moved_to.r, moved_to.s)
        if (moved_to_value != ChineseCheckers.EMPTY_SPACE):
            return False
        
        # If an action is a jump, there must be a direct neighbor between the
        # start position and the moved to position.
        if action.is_jump:
            direct_neighbor = action.position.neighbor(action.direction)
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
    
    def is_action_legal(self, player: int, action: ChineseCheckersAction) -> bool:
        self._set_rotation(player)
        is_legal = self._is_action_legal(player, action)
        self._set_rotation(0)
        return is_legal
    
    def move(self, player: int, action: ChineseCheckersAction):
        assert self.is_action_legal(player, action)
        self._set_rotation(player)
        src_pos = action.position
        self._set_coordinate(src_pos.q, src_pos.r, src_pos.s, ChineseCheckers.EMPTY_SPACE)
        dst_pos = action.moved_position()
        self._set_coordinate(dst_pos.q, dst_pos.r, dst_pos.s, player)
        self._set_rotation(0)
        
class raw_env(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "chinese_checkers"}

    def __init__(self, triangle_size: int, max_iters: int = 200):
        self.max_iters = max_iters

        # Players 0 through 5 are the six players
        self.possible_agents = [f"player_{r}" for r in range(6)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.window_size = 512  # The size of the PyGame window

        self.n = triangle_size
        self.rotation = 0
        self.game = ChineseCheckers(triangle_size)
        self.game.init_board()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.game.init_board()
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        player: int = self.agent_name_mapping[agent]

        if not self.game.is_action_legal(player, action):
            raise Exception("Played an illegal move.")

        self.game.move(player, action)

        # TODO: Check game over, set terminations, rewards, etc.

        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_iters for agent in self.agents
            }

            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        return self.game.render()

    def observation_space(self, agent):
        """ The observation is just the board, regardless of agent.
            To counteract the sparsity of cube coordinates, we convert it to axial
                (q, r, s) -> (q, r)
            first.
            -2: invalid space
            -1: empty space
            0-5: player number of occupying peg
        """
        return self.observe(agent)

    def observe(self, agent):
        """ The observation is just the board, regardless of agent.
            To counteract the sparsity of cube coordinates, we convert it to axial
                (q, r, s) -> (q, r)
            first.
            -2: invalid space
            -1: empty space
            0-5: player number of occupying peg
        """
        return Box(low=-2, high=5, shape=(4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)

    def action_space(self, agent):
        """ (4 * n + 1)^2 spaces in the (axial) board, 6 directions to move for each, 2 types (no-jump/jump)
        
        Action is a tuple of (q, r, direction, is_jump), where
        
        q, r: axial coordinates of the peg to move
        direction: 0-5, the direction to move in, similar to polar coordinate increments of pi/3, where 0 is right, 1 is top-right, etc.
        is_jump: 0 or 1, whether to jump over a peg
        
        """
        return MultiDiscrete([4 * self.n + 1, 4 * self.n + 1, 6, 2], dtype=np.int8)

if __name__ == "__main__":
    env = raw_env(4)
    env.reset()
    board = env.observe(0)
    env.game.move(0, ChineseCheckersAction(2, -6, 4, True))
    frame = env.render()
    plt.imshow(frame)
    plt.savefig("test.png")

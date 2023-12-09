import functools
import re

import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple, MultiBinary

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import pygame
from chinese_checkers.env.game import ChineseCheckers, Move, Position
from chinese_checkers.env.chinese_checkers_utils import action_to_move, get_legal_move_mask

def make_env(triangle_size=4):
    env = raw_env(triangle_size)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
        
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
        self.action_space_dim = (4 * self.n + 1) * (4 * self.n + 1) * 6 * 2

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.game.init_game()
        self.observations = {
            agent: self.game.get_axial_board(self.agent_name_mapping[agent]) for agent in self.agents
        }
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent: str = self.agent_selection
        player: int = self.agent_name_mapping[agent]

        action = int(action)

        move = action_to_move(action, self.n)
        score = self.game.move(player, move)

        # If the current player wins, set their reward to 5N and other agents to -N
        # Otherwise, set the reward to the number of pegs that the player has in the target zone
        self._cumulative_rewards[agent] = 0
        if self.game.did_player_win(self.agent_name_mapping[agent]):
            self.terminations = {
                agent: self.game.is_game_over() for agent in self.agents
            }
            for a in self.agents:
                self.rewards[a] = self.n * 5 if a == agent else -self.n
        else:
            self.rewards[agent] = score
        self._accumulate_rewards()
        self._clear_rewards()

        if self._agent_selector.is_last():
            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_iters for agent in self.agents
            }
            
        # if jump then reorder the agent selector so the current player is the next player
        # if not self.game.last_move.is_jump:
        #     self.agent_selection = self._agent_selector.next()
        self.agent_selection = self._agent_selector.next()


    def render(self):
        return self.game.render()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """ The observation is the board from the current player's perspective.
            To counteract the sparsity of cube coordinates, we convert it to axial
                (q, r, s) -> (q, r)
            first.
            -2: invalid space
            -1: empty space
            0-5: player number of occupying peg
        """
        return {
            "observation": Box(low=-2, high=5, shape=(4 * self.n + 1, 4 * self.n + 1)),
            "action_mask": Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.int8)
        }
        
    def observe(self, agent):
        """ The observation is just the board with the current player at the top, regardless of agent.
            To counteract the sparsity of cube coordinates, we convert it to axial
                (q, r, s) -> (q, r)
            first.
            -2: invalid space
            -1: empty space
            0-5: player number of occupying peg
        """
        return {
            "observation": self.game.get_axial_board(self.agent_name_mapping[agent]),
            "action_mask": get_legal_move_mask(self.game, self.agent_name_mapping[agent])
        }

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """ (4 * n + 1)^2 spaces in the (axial) board, 6 directions to move for each, 2 types (no-jump/jump)
        
        Action is a tuple of (q, r, direction, is_jump), where
        
        q, r: axial coordinates of the peg to move
        direction: 0-5, the direction to move in, similar to polar coordinate increments of pi/3, where 0 is right, 1 is top-right, etc.
        is_jump: 0 or 1, whether to jump over a peg
        
        """
        # return Tuple([Discrete(4 * self.n + 1, start=-2 * self.n),
        #               Discrete(4 * self.n + 1, start=-2 * self.n),
        #               Discrete(6),
        #               Discrete(2)])

        return Discrete(self.action_space_dim)
        

if __name__ == "__main__":
    env = raw_env(4)
    env.reset()
    # board = env.observe(0)
    board = env.game.get_axial_board(0)
    # env.game.move(0, Move(2, -6, 4, True))

    frame = env.render()
    plt.imshow(frame)
    plt.savefig("test.png")

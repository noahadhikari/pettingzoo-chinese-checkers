import functools
import re

import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from gymnasium.spaces import Box, Discrete, Dict

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import pygame
from chinese_checkers.env.game import ChineseCheckers, Direction, Move, Position
from chinese_checkers.env.chinese_checkers_utils import action_to_move, get_legal_move_mask, rotate_observation

def env(**kwargs):
    env = raw_env(**kwargs)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
        
class raw_env(AECEnv):
    metadata = {"render_modes": ["rgb_array", "human"], "name": "chinese_checkers"}

    def __init__(self, render_mode: str = "rgb_array", triangle_size: int = 4, max_iters: int = 1000, **kwargs):
        self.max_iters = max_iters

        # Players 0 through 5 are the six players
        self.agents = [f"player_{r}" for r in range(6)]
        self.possible_agents = self.agents[:]   
        self._agent_selector = agent_selector(self.agents)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        self.n = triangle_size
        self.iters = 0
        self.num_moves = 0
        self.winner = None

        self.rewards = None
        self.infos = {agent: {} for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}

        self.action_space_dim = (4 * self.n + 1) * (4 * self.n + 1) * 6 * 2 + 1
        self.observation_space_dim = (4 * self.n + 1) * (4 * self.n + 1) * 8
        self.action_spaces = {agent: Discrete(self.action_space_dim) for agent in self.agents}
        self.observation_spaces = {
            agent: Dict({
                # "observation": Box(low=0, high=1, shape=(4 * self.n + 1, 4 * self.n + 1, 8)),
                "observation": Box(low=0, high=1, shape=(self.observation_space_dim,)),
                "action_mask": Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.int8)
            })
            for agent in self.agents
        }

        self.agent_selection = None

        self.window_size = 512  # The size of the PyGame window
        self.render_mode = render_mode

        self.rotation = 0
        self.game = ChineseCheckers(triangle_size, render_mode=render_mode)

    def reset(self, seed=None, return_info=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.game.init_game()
        self.iters = 0
        self.num_moves = 0
        self.winner = None

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
        move = self.game.move(player, move)

        # If the current player wins, set their reward to 5N and other agents to -N
        # Otherwise, set the reward to the number of pegs that the player has in the target zone
        # self._cumulative_rewards[agent] = 0
        if self.game.did_player_win(self.agent_name_mapping[agent]):
            self.terminations = {
                agent: self.game.is_game_over() for agent in self.agents
            }
            for a in self.agents:
                self.rewards[a] = 10 if a == agent else -1
            self.winner = agent
        elif move is None:
                self.rewards[agent] = -1000
        else:            
            if isinstance(move, Move) and move.direction in [Direction.DownLeft, Direction.DownRight]:
                self.rewards[agent] = 0.001
            if isinstance(move, Move) and move.direction in [Direction.UpLeft, Direction.UpRight]:
                self.rewards[agent] = -0.001
            if move and move != Move.END_TURN:
                src_pos = move.position
                dst_pos = move.moved_position()
                target = [Position(q, r) for q, r, s in self.game.get_target_coordinates(player)]
                if src_pos not in target and dst_pos in target:
                    self.rewards[agent] += 0.1
                if src_pos in target and dst_pos not in target:
                    self.rewards[agent] -= 0.1

        self._accumulate_rewards()
        self._clear_rewards()

        if self._agent_selector.is_last():
            self.truncations = {
                agent: self.iters >= self.max_iters for agent in self.agents
            }
        self.num_moves += 1

        # if jump then don't advance the current player
        if move == Move.END_TURN or not (move and move.is_jump):
            self.agent_selection = self._agent_selector.next()
        elif self._agent_selector.is_last():
            self.iters += 1

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        else:
            return self.game.render()

    def observation_space(self, agent):
        """ See `observe` method for details on the observation space.
        """
        return self.observation_spaces[agent]
        
    def observe(self, agent):
        """ 
        The observation is a (4n + 1) x (4n + 1) board with 8 channels. The 8 channels correspond to:
        - Channels 0: Current player's pieces, 1 if that position has the current player's peg, 0 if not
        - Channels 1-5: Other player's pieces
        - Channel 6: Source pegs of all previous jumps for this player. If no jumps, then all zeros.
        - Channel 7: Last jump destination peg: If no jumps, then all zeros.
        """
        player = self.agent_name_mapping[agent]
        board = self.game.get_axial_board(player)

        jump_sources_channel = -2 * np.zeros((4 * self.game.n + 1, 4 * self.game.n + 1), dtype=np.int8)
        last_jump_destination_channel = -2 * np.zeros((4 * self.game.n + 1, 4 * self.game.n + 1), dtype=np.int8)
        
        last_jump = self.game.get_last_jump(player)
        if last_jump:
            jumps = self.game.get_jumps(player)
            for jump in jumps:
                jump_sources_channel[jump.position.q, jump.position.r] = 1
            last_jump_destination = last_jump.moved_position()
            last_jump_destination_channel[last_jump_destination.q, last_jump_destination.r] = 1

        observation = np.stack(
            [(board == player).astype(np.int8) for player in range(6)] + 
            [jump_sources_channel, last_jump_destination_channel],
            axis=-1
        )

        observation = observation.flatten()

        return {
            "observation": observation,
            "action_mask": get_legal_move_mask(self.game, player)
        }
    
    # def action_mask(self):
    #     agent: str = self.agent_selection
    #     player: int = self.agent_name_mapping[agent]
    #     return get_legal_move_mask(self.game, player)
    
    def action_space(self, agent):
        """ (4 * n + 1)^2 spaces in the (axial) board, 6 directions to move for each, 2 types (no-jump/jump) + 1 no-op
        
        Action is a tuple of (q, r, direction, is_jump), where
        
        q, r: axial coordinates of the peg to move
        direction: 0-5, the direction to move in, similar to polar coordinate increments of pi/3, where 0 is right, 1 is top-right, etc.
        is_jump: 0 or 1, whether to jump over a peg
        
        """
        return self.action_spaces[agent]
    
    def close(self):
        pass

if __name__ == "__main__":
    env = raw_env(4)
    env.reset()
    # board = env.observe(0)
    board = env.game.get_axial_board(0)
    # env.game.move(0, Move(2, -6, 4, True))

    frame = env.render()
    plt.imshow(frame)
    plt.savefig("test.png")

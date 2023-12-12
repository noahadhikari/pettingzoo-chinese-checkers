import datetime
import os
import numpy as np
import glob
import argparse

from gymnasium.spaces import Box, Discrete

from pettingzoo.classic import rps_v2

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy

from chinese_checkers import chinese_checkers_v0
from chinese_checkers.models.action_masking_rlm import TorchActionMaskRLM
from chinese_checkers.models.action_masking import ActionMaskModel
from chinese_checkers.scripts.logger import custom_log_creator

def eval(triangle_size: int = 4, policy_name: str = "default_policy", num_games: int = 2, against_self: bool = False, checkpoint_path: str = None):
    # Evaluate a trained agent vs a random agent

    # Use the `from_checkpoint` utility of the Policy class:
    policy = Policy.from_checkpoint(checkpoint_path)
    policy = policy[policy_name]

    # Use the restored policy for serving actions.
    if against_self:
        evaluate_policies(policy, policy, triangle_size=triangle_size, num_games=num_games)
    else:
        evaluate_policy_against_random(policy, triangle_size=triangle_size, num_games=num_games)
    

def evaluate_policies(eval_policy, baseline_policy, triangle_size=4, num_games=2):
    """
    Evaluate two policies against one another. eval_policy will play as player 0, baseline_policy will play as player 1-5.
    """
    env = chinese_checkers_v0.env(render_mode="human", triangle_size=triangle_size)
    print(
        f"Starting evaluation of {eval_policy} against baseline {baseline_policy}. Trained agent will play as {env.possible_agents[0]}."
    )

    total_rewards = {agent: 0 for agent in env.possible_agents}
    wins = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        for a in range(6):
            env.action_space(env.possible_agents[a]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            total_rewards[agent] += reward
            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    action = eval_policy.compute_single_action(obs)
                else:
                    action = baseline_policy.compute_single_action(obs)
            act = int(action[0])
            env.render()
            env.step(act)

        # accumulate rewards after game ends
        for agent in env.possible_agents:
            rew = env._cumulative_rewards[agent]
            total_rewards[agent] += rew
            if rew == 10:
                wins[agent] += 1
                
    env.close()

    winrate = wins[env.possible_agents[0]] / num_games
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Average rewards (incl. negative rewards): ", {agent: total_rewards[agent] / num_games for agent in env.possible_agents})
    print("Winrate: ", winrate)


class ChineseCheckersRandomPolicy(Policy):
    def __init__(self, triangle_size=4, config={}):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 8,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.action_space = action_space

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            action = self.action_space.sample(obs["action_mask"])
            actions.append(action)
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None, prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions([obs], state_batches=[state], prev_action_batch=[prev_action], prev_reward_batch=[prev_reward], info_batch=[info], episodes=[episode], **kwargs)[0]


def evaluate_policy_against_random(policy, triangle_size=4, num_games=2):
    return evaluate_policies(policy, ChineseCheckersRandomPolicy(triangle_size), triangle_size, num_games)

def main(args):
    # define how to make the environment. This way takes an optional environment config, num_floors
    env_creator = lambda config: chinese_checkers_v0.env(
        triangle_size=config["triangle_size"]
    )

    # register that way to make the environment under an rllib name
    env_name = 'chinese_checkers_v0'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    ray.init(num_cpus=1 or None, local_mode=True)
    if args.eval_random:
        eval(triangle_size=args.triangle_size, policy_name=args.policy_name, against_self=False, checkpoint_path=args.checkpoint_path)
    elif args.eval_self:
        eval(triangle_size=args.triangle_size, policy_name=args.policy_name, against_self=True, checkpoint_path=args.checkpoint_path)
    else:
        print("Did not specify train or eval.")
        return
    print("Finished successfully without selecting invalid actions.")
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RLLib train/eval script'
    )
    parser.add_argument('-er', '--eval_random',
                        action='store_true')  # on/off flag
    parser.add_argument('-es', '--eval_self',
                        action='store_true')  # on/off flag
    parser.add_argument('-p', '--policy_name', default="default_policy")
    parser.add_argument('-s', '--triangle_size', type=int, required=True)
    parser.add_argument('-c', '--checkpoint_path', type=str, required=False)
    args = parser.parse_args()
    main(args)
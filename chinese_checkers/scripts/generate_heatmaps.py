import csv
import datetime
import os
import numpy as np
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

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

def write_to_csv(filename, results):
    path_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not path_exists:
            writer.writerow(list(results.keys()))
        writer.writerow(list(results.values()))

def train(config, model_name: str, train_config):
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/chinese_checkers_{model_name}_{timestr}"
    algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, logdir), ''))

    triangle_size = train_config["triangle_size"]
    train_iters = train_config["train_iters"]
    eval_period = train_config["eval_period"]
    print(eval_period)
    eval_config = {
        "triangle_size": train_config["triangle_size"],
        "eval_num_trials": train_config["eval_num_trials"],
        "eval_max_iters": train_config["eval_max_iters"],
        "render_mode": train_config["render_mode"],
        "logdir": logdir
    }

    # run manual training loop and print results after each iteration
    for i in range(train_iters):
        result = algo.train()
        checkpoint_dir = f"{logdir}/checkpoint{i}"
        save_result = algo.save(checkpoint_dir=checkpoint_dir)
        path_to_checkpoint = save_result.checkpoint.path
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )
        # print(pretty_print(result))
        print(f"""
              Iteration {i}: episode_reward_mean = {result['episode_reward_mean']},
                             episode_reward_max  = {result['episode_reward_max']},
                             episode_reward_min  = {result['episode_reward_min']},
                             episode_len_mean    = {result['episode_len_mean']}
              """)

        train_results = {
            "iteration": i,
            "env_steps": result["num_env_steps_sampled"],
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_reward_min": result["episode_reward_min"],
            "episode_len_mean": result["episode_len_mean"]
        } 
        for policy_id in range(6):
            if f"policy_{policy_id}" in result["policy_reward_mean"]:
                train_results[f"policy_{policy_id}_reward_mean"] = result["policy_reward_mean"][f"policy_{policy_id}"]
            else:
                train_results[f"policy_{policy_id}_reward_mean"] = np.nan

        train_logs_path = f"{logdir}/train_logs.csv"
        write_to_csv(train_logs_path, train_results)
        
        policy = None
        if algo.get_policy("default_policy"):
            policy = algo.get_policy("default_policy")
        else:
            policy = algo.get_policy("policy_0")
        if i % eval_period == 0:
            eval_results = { "iteration": i } | evaluate_policy_against_random(policy, eval_config)
            eval_logs_path = f"{logdir}/eval_logs.csv"
            write_to_csv(eval_logs_path, eval_results)

        
    return algo

def eval(checkpoint_path: str = None, eval_config=None):
    # Evaluate a trained agent vs a random agent

    # Use the `from_checkpoint` utility of the Policy class:
    policy = Policy.from_checkpoint(checkpoint_path)
    print(policy)
    if "default_policy" in policy:
        policies = {f"player_{i}": policy["default_policy"] for i in range(6)}
    else:
        policies = {f"player_{i}": policy[f"policy_{i}"] for i in range(6)}
        
    evaluate_policies(policies, eval_config)

def evaluate_policies(policies, eval_config):
    """
    Evaluate two policies against one another. eval_policy will play as player 0, baseline_policy will play as player 1-5.
    """

    triangle_size = eval_config["triangle_size"]
    eval_num_trials = eval_config["eval_num_trials"]
    eval_max_iters = eval_config["eval_max_iters"]
    render_mode = eval_config["render_mode"]

    env = chinese_checkers_v0.env(render_mode=render_mode, triangle_size=triangle_size, max_iters=eval_max_iters)

    total_rewards = {agent: 0 for agent in env.possible_agents}
    wins = {agent: 0 for agent in env.possible_agents}
    num_moves = []

    boards = []

    for i in tqdm(range(eval_num_trials)):
        env.reset(seed=i)
        for a in range(6):
            env.action_space(env.possible_agents[a]).seed(i)

        num_moves = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            total_rewards[agent] += reward
            if termination or truncation:
                break
            else:
                action = policies[agent].compute_single_action(obs)
            act = int(action[0])
            if render_mode:
                env.render()

            if num_moves == 10:
                boards.append(env.unwrapped.game.get_axial_board(0))
            num_moves += 1
            env.step(act)
    
    with open("boards.npy", "wb") as f:
        np.save(f, np.array(boards))
    env.close()

    winrate = wins[env.possible_agents[0]] / eval_num_trials
    average_rewards = {agent: total_rewards[agent] / eval_num_trials for agent in env.possible_agents}
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Average rewards (incl. negative rewards): ", average_rewards)
    print("Winrate: ", winrate)
    print("Average moves:", np.mean(num_moves))

    return {
        "eval_num_trials": eval_num_trials,
        "eval_total_rewards": total_rewards["player_0"],
        "eval_average_rewards": average_rewards["player_0"],
        "eval_win_rate": winrate,
        "eval_average_moves": np.mean(num_moves)
    }

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


def evaluate_policy_against_random(policy, eval_config):
    triangle_size = eval_config["triangle_size"]
    return evaluate_policies(policy, ChineseCheckersRandomPolicy(triangle_size), eval_config)

def main(args):
    # define how to make the environment. This way takes an optional environment config, num_floors
    env_creator = lambda config: chinese_checkers_v0.env(
        triangle_size=config["triangle_size"]
    )

    # register that way to make the environment under an rllib name
    env_name = 'chinese_checkers_v0'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    ray.init(num_cpus=1 or None, local_mode=True)
    eval_config = {
        "triangle_size": args.triangle_size,
        "eval_max_iters": args.eval_max_iters,
        "eval_num_trials": args.eval_num_trials,
        "render_mode": args.render_mode
    }

    eval(args.checkpoint_path, eval_config)
    
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
    parser.add_argument('--triangle_size', type=int, required=True)
    parser.add_argument('--eval_num_trials', type=int, default=10)
    parser.add_argument('--eval_max_iters', type=int, default=300)
    parser.add_argument('--checkpoint_path', type=str, required=False)
    parser.add_argument('--render_mode', default=None)
    args = parser.parse_args()
    main(args)
import numpy as np
import argparse
from tqdm import tqdm
import os
import glob

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from ray.rllib.policy.policy import Policy

from chinese_checkers import chinese_checkers_v0

import matplotlib.pyplot as plt

import json
import jsonpickle

def eval(checkpoints: str, eval_config=None):

    # Use the `from_checkpoint` utility of the Policy class:
    # policies = {}
    # for i in range(100):
    #     policy = Policy.from_checkpoint(f"{checkpoint}/checkpoint{i}")
    #     if "default_policy" in policy:
    #         policies[f"player_{i}"] = policy["default_policy"]
    #     else:
    #         policies[f"player_{i}"] = policy[f"policy_{i}"]
    # evaluate_policies(policies, eval_config)

    # iterate over each subfolder in checkpoints, and evaluate the policy in that subfolder, from checkpoint0 to checkpoint99

    # get the subfolders under checkpoints
    independent = "independent_n2"
    shared_encoder = "shared_encoder_n2"
    fully_shared = "full_sharing_n2"

    checkpoint_start = eval_config["checkpoint_start"]
    checkpoint_end = eval_config["checkpoint_end"]

    raw_results = {i: [] for i in range(checkpoint_start, checkpoint_end)}

    num_trials_per_iter = eval_config["eval_num_trials"]

    eval_config["eval_num_trials"] = 1 # only need to evaluate once per random shuffle

    for i in range(checkpoint_start, checkpoint_end):
        print(f"evaluating checkpoint {i}")
        results = []
        for j in range(num_trials_per_iter):
            policies = {}

            # assign two random indices between 0 and 6 to indpendent, shared_encoder, and fully_shared, without replacement
            indices = np.arange(6)
            np.random.shuffle(indices)
            independent_indices = indices[:2]
            shared_encoder_indices = indices[2:4]
            fully_shared_indices = indices[4:]

            player_map = {}
            for idx in independent_indices:
                player_map[idx] = independent
            for idx in shared_encoder_indices:
                player_map[idx] = shared_encoder
            for idx in fully_shared_indices:
                player_map[idx] = fully_shared

            for k in range(6):
                pol = player_map[k]
                policy = Policy.from_checkpoint(f"{checkpoints}/{pol}/checkpoint{i}")
                if "default_policy" in policy:
                    policies[f"player_{k}"] = policy["default_policy"]
                else:
                    policies[f"player_{k}"] = policy[f"policy_{k}"]
            
            result = evaluate_policies(policies, eval_config)
            result["players"] = player_map

            raw_results[i].append(result)

            # save the raw results to a file
            with open("raw_results2.json", "w") as f:
                json.dump(jsonpickle.encode(raw_results), f, indent=4)

    # plot the results as a line plot over the number of iterations, with one line for each policy
    # plt.plot(results[independent], label="independent")
    # plt.plot(results[shared_encoder], label="shared_encoder")
    # plt.plot(results[fully_shared], label="fully_shared")

    # # plot game lengths on another vertical axis on the same graph
    # # ax2 = plt.twinx()
    # # ax2.plot(game_lengths, label="game_lengths", color="black")

    # plt.legend()
    # plt.savefig("results.png")


def evaluate_policies(policies, eval_config):
    """
    Evaluate several policies against one another.
    Policies should be a dict of {agent_name: policy}
    Eval config should be a dict of {eval_config_name: eval_config_value}
    """

    triangle_size = eval_config["triangle_size"]
    eval_num_trials = eval_config["eval_num_trials"]
    eval_max_iters = eval_config["eval_max_iters"]
    render_mode = eval_config["render_mode"]

    env = chinese_checkers_v0.env(render_mode=render_mode, triangle_size=triangle_size, max_iters=eval_max_iters)

    total_rewards = {agent: 0 for agent in env.possible_agents}
    wins = {agent: 0 for agent in env.possible_agents}
    game_lengths = []
    wins["truncated"] = 0

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
            num_moves += 1
            env.step(act)

        # accumulate rewards after game ends
        for agent in env.possible_agents:
            rew = env._cumulative_rewards[agent]
            total_rewards[agent] += rew

        if env.unwrapped.winner:
            wins[env.unwrapped.winner] += 1
        else:
            wins["truncated"] += 1

        game_lengths.append(num_moves)

    winrates = {agent: wins[agent] / eval_num_trials for agent in env.possible_agents}
    winrates["truncated"] = wins["truncated"] / eval_num_trials
    average_rewards = {agent: total_rewards[agent] / eval_num_trials for agent in env.possible_agents}
    # print("Total rewards (incl. negative rewards): ", total_rewards)
    # print("Average rewards (incl. negative rewards): ", average_rewards)
    # print("Winrates: ", winrates)
    # print("Average moves:", mean_num_moves)

    return {
        "eval_num_trials": eval_num_trials,
        "eval_total_rewards": total_rewards,
        "eval_average_rewards": average_rewards,
        "eval_win_rates": winrates,
        "eval_game_lengths": game_lengths
    }



def main(args):
    # define how to make the environment. This way takes an optional environment config, num_floors
    env_creator = lambda config: chinese_checkers_v0.env(
        triangle_size=config["triangle_size"]
    )

    # register that way to make the environment under an rllib name
    env_name = 'chinese_checkers_v0'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    ray.init(num_cpus=4 or None, local_mode=True)
    eval_config = {
        "triangle_size": args.triangle_size,
        "eval_max_iters": args.eval_max_iters,
        "eval_num_trials": args.eval_num_trials,
        "render_mode": args.render_mode,
        "checkpoint_start": args.checkpoint_start,
        "checkpoint_end": args.checkpoint_end
    }

    eval(os.path.join(os.curdir, "logs"), eval_config)
    
    print("Finished successfully without selecting invalid actions.")
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RLLib train/eval script'
    )
    parser.add_argument('--triangle_size', type=int, required=True)
    parser.add_argument('--eval_num_trials', type=int, default=10)
    parser.add_argument('--eval_max_iters', type=int, default=300)
    parser.add_argument('--render_mode', default=None)
    parser.add_argument('--checkpoint_start', type=int, default=0)
    parser.add_argument('--checkpoint_end', type=int, default=100)
    args = parser.parse_args()
    main(args)
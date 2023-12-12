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

def train(env_name: str, triangle_size: int = 4):
    rlm_class = TorchActionMaskRLM

    model_config = {
        "fcnet_hiddens": [64, 64]
    }

    rlm_spec = SingleAgentRLModuleSpec(
        module_class=rlm_class, 
        model_config_dict=model_config
    )

    action_space_dim = (4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1
    # observation_space_shape = (4 * triangle_size + 1, 4 * triangle_size + 1, 8)
    observation_space_shape = ((4 * triangle_size + 1) * (4 * triangle_size + 1) * 8,)

    # main part: configure the ActionMaskEnv and ActionMaskModel
    config = (
        PPOConfig()
        .environment(
            # random env with 100 discrete actions and 5x [-1,1] observations
            # some actions are declared invalid and lead to errors
            env=env_name, 
            clip_actions=True,
            env_config={
                "triangle_size": triangle_size,
                "action_space": Discrete(action_space_dim),
                "max_iters": 800,
                "render_mode": None,
                # This is not going to be the observation space that our RLModule sees.
                # It's only the configuration provided to the environment.
                # The environment will instead create Dict observations with
                # the keys "observation" and "action_mask".
                "observation_space": Box(low=0, high=1, shape=(observation_space_shape), dtype=np.int8),
            },
        )
        .training(
            # train_batch_size=512,
            # lr=2e-5,
            # gamma=0.99,
            # lambda_=0.9,
            # use_gae=True,
            # clip_param=0.4,
            # grad_clip=None,
            # entropy_coeff=0.1,
            # vf_loss_coeff=0.25,
            # sgd_minibatch_size=64,
            # num_sgd_iter=10,
        )
        # We need to disable preprocessing of observations, because preprocessing
        # would flatten the observation dict of the environment.
        .experimental(
            # _enable_new_api_stack=True,
            _disable_preprocessor_api=True,
        )
        .framework("torch")
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        )
        .rl_module(rl_module_spec=rlm_spec)
    )
    print(config.to_dict())
    
    algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, "logs"), ''))

    # run manual training loop and print results after each iteration
    for i in range(100):
        result = algo.train()
        timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = "{}_{}".format("checkpoints/chinese_checkers", timestr)
        save_result = algo.save(checkpoint_dir=logdir)
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
    evaluate_policy_against_random(algo.get_policy(), triangle_size=triangle_size)
    return algo

def eval(triangle_size: int = 4, num_games: int = 2):
    # Evaluate a trained agent vs a random agent

    # Use the `from_checkpoint` utility of the Policy class:
    checkpoint_path = "checkpoints/chinese_checkers_2023-12-11_20-50-16"
    policy = Policy.from_checkpoint(checkpoint_path)
    policy = policy['default_policy']

    # Use the restored policy for serving actions.
    evaluate_policy_against_itself(policy, triangle_size=triangle_size)
    

def evaluate_policy_against_random(policy, triangle_size=4, num_games=2):
    env = chinese_checkers_v0.env(render_mode="human", triangle_size=triangle_size)
    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[0]}."
    )

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    action = policy.compute_single_action(obs)
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(
                        action[0]
                    )
                else:
                    act = env.action_space(agent).sample(action_mask)
            env.render()
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores

def evaluate_policy_against_itself(policy, triangle_size=4, num_games=2):
    env = chinese_checkers_v0.env(render_mode="human", triangle_size=triangle_size)
    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[0]}."
    )

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                break
            else:
                action = policy.compute_single_action(obs)
                # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                act = int(
                    action[0]
                )
            env.render()
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores

def main(args):
    # define how to make the environment. This way takes an optional environment config, num_floors
    env_creator = lambda config: chinese_checkers_v0.env(
        triangle_size=config["triangle_size"],
        max_iters=config["max_iters"],
        render_mode=config["render_mode"]
    )

    # register that way to make the environment under an rllib name
    env_name = 'chinese_checkers_v0'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    ray.init(num_cpus=1 or None, local_mode=True)
    if args.train:
        algo = train(env_name, triangle_size=args.triangle_size)
    elif args.eval_random:
        eval(triangle_size=args.triangle_size)
    elif args.eval_self:
        eval(triangle_size=args.triangle_size)
    else:
        print("Did not specify train or eval.")
        return
    print("Finished successfully without selecting invalid actions.")
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RLLib train/eval script'
    )
    parser.add_argument('--eval_random',
                        action='store_true')  # on/off flag
    parser.add_argument('--eval_self',
                        action='store_true')  # on/off flag
    parser.add_argument('--train',
                        action='store_true')  # on/off flag
    parser.add_argument('-s', '--triangle_size', type=int, required=True)
    args = parser.parse_args()
    main(args)
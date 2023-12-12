import os
import numpy as np
from chinese_checkers.models.action_masking_rlm import TorchActionMaskRLM
from gymnasium.spaces import Box, Discrete
import ray
from ray import tune
from ray.tune.registry import register_env
from pettingzoo.classic import rps_v2
from chinese_checkers import chinese_checkers_v0
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.logger import pretty_print

from chinese_checkers.models.action_masking import ActionMaskModel
from chinese_checkers.scripts.logger import custom_log_creator

ModelCatalog.register_custom_model(
    "pa_model", ActionMaskModel
)

# define how to make the environment. This way takes an optional environment config, num_floors
env_creator = lambda config: chinese_checkers_v0.env(
    triangle_size=config["triangle_size"]
)

# register that way to make the environment under an rllib name
env_name = 'chinese_checkers_v0'
register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

triangle_size = 2
test_env = PettingZooEnv(env_creator({"triangle_size": triangle_size}))
# test_env = MultiAgentEnvCompatibility(test_env)
obs_space = test_env.observation_space
act_space = test_env.action_space

ray.init(num_cpus=1 or None, local_mode=True)

rlm_class = TorchActionMaskRLM
rlm_spec = SingleAgentRLModuleSpec(
    module_class=rlm_class,
    # observation_space=test_env.observation_space,
    # action_space=test_env.action_space,
)

action_space_dim = (4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1
observation_space_dim = (4 * triangle_size + 1, 4 * triangle_size + 1, 8)

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
            # This is not going to be the observation space that our RLModule sees.
            # It's only the configuration provided to the environment.
            # The environment will instead create Dict observations with
            # the keys "observation" and "action_mask".
            "observation_space": Box(low=0, high=1, shape=observation_space_dim, dtype=np.int8),
        },
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

algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, "logs"), ''))

# run manual training loop and print results after each iteration
for _ in range(100):
    result = algo.train()
    print(pretty_print(result))

# # manual test loop
# print("Finished training. Running manual test/inference loop.")
# # prepare environment with max 10 steps
# config["env_config"]["max_episode_len"] = 10
# env = ActionMaskEnv(config["env_config"])
# obs, info = env.reset()
# done = False
# # run one iteration until done
# print(f"ActionMaskEnv with {config['env_config']}")
# while not done:
#     action = algo.compute_single_action(obs)
#     next_obs, reward, done, truncated, _ = env.step(action)
#     # observations contain original observations and the action mask
#     # reward is random and irrelevant here and therefore not printed
#     print(f"Obs: {obs}, Action: {action}")
#     obs = next_obs

print("Finished successfully without selecting invalid actions.")
ray.shutdown()
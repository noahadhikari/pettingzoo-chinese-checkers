import os
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
# from ray.rllib.models import ModelCatalog

# from chinese_checkers.models.action_masking import ActionMaskModel

# ModelCatalog.register_custom_model(
#     "pa_model", ActionMaskModel
# )

# define how to make the environment. This way takes an optional environment config, num_floors
env_creator = lambda config: chinese_checkers_v0.env(
    triangle_size=config["triangle_size"]
)

# env_creator = lambda config: rps_v2.env()

# register that way to make the environment under an rllib name
env_name = 'chinese_checkers_v0'
register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

ray.init()

test_env = PettingZooEnv(env_creator({"triangle_size": 4}))
# test_env = MultiAgentEnvCompatibility(test_env)
obs_space = test_env.observation_space
act_space = test_env.action_space

config = (
    PPOConfig()
    .environment(env=env_name, clip_actions=True)
    .rollouts(num_rollout_workers=1, rollout_fragment_length=128)
    .training(
        train_batch_size=512,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=64,
        num_sgd_iter=10,
        _enable_learner_api=False
    )
    .rl_module(_enable_rl_module_api=False)
    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)
# you can pass arguments to the environment creator with the env_config option in the config
config['env_config'] = {"triangle_size": 2}
config["model"]["custom_model"] = "pa_model"

tune.run(
    "PPO",
    name="PPO",
    stop={"timesteps_total": 10000 if not os.environ.get("CI") else 50000},
    checkpoint_freq=10,
    local_dir="~/ray_results/" + env_name,
    config=config,
)
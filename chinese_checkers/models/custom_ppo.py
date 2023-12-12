# https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html#extending-existing-rllib-rl-modules

import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.configs import MLPHeadConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.examples.models.mobilenet_v2_encoder import (
    MobileNetV2EncoderConfig,
    MOBILENET_INPUT_SHAPE,
)
from ray.rllib.core.models.configs import ActorCriticEncoderConfig

class MobileNetTorchPPORLModule(PPOTorchRLModule):
    """A PPORLModules with mobilenet v2 as an encoder.

    The idea behind this model is to demonstrate how we can bypass catalog to
    take full control over what models and action distribution are being built.
    In this example, we do this to modify an existing RLModule with a custom encoder.
    """

    def setup(self):
        mobilenet_v2_config = MobileNetV2EncoderConfig()
        # Since we want to use PPO, which is an actor-critic algorithm, we need to
        # use an ActorCriticEncoderConfig to wrap the base encoder config.
        actor_critic_encoder_config = ActorCriticEncoderConfig(
            base_encoder_config=mobilenet_v2_config
        )

        self.encoder = actor_critic_encoder_config.build(framework="torch")
        mobilenet_v2_output_dims = mobilenet_v2_config.output_dims

        pi_config = MLPHeadConfig(
            input_dims=mobilenet_v2_output_dims,
            output_layer_dim=2,
        )

        vf_config = MLPHeadConfig(
            input_dims=mobilenet_v2_output_dims, output_layer_dim=1
        )

        self.pi = pi_config.build(framework="torch")
        self.vf = vf_config.build(framework="torch")

        self.action_dist_cls = TorchCategorical
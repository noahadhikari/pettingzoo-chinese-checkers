import json
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.configs import (ActorCriticEncoderConfig,
                                           MLPHeadConfig)
from ray.rllib.core.rl_module.marl_module import (MultiAgentRLModule,
                                                  MultiAgentRLModuleConfig)
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.examples.models.mobilenet_v2_encoder import (
    MOBILENET_INPUT_SHAPE, MobileNetV2EncoderConfig)
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.nested_dict import NestedDict
from chinese_checkers.models.action_masking_rlm import TorchActionMaskRLM
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel

class SharedEncoderConfig(ModelConfig):
    def __init__(self, input_dim=None, hidden_dims=None, freeze=False):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.freeze = freeze

    def build(self, framework):
        assert framework == "torch", "Unsupported framework `{}`!".format(framework)
        return SharedEncoder(self)


class SharedEncoder(TorchModel, Encoder):
    """A MobileNet v2 encoder for RLlib."""

    def __init__(self, config):
        super().__init__(config)
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
        )
        if config.freeze:
            # We don't want to train this encoder, so freeze its parameters!
            for p in self.net.parameters():
                p.requires_grad = False

    def _forward(self, input_dict, **kwargs):
        return {ENCODER_OUT: (self.net(input_dict["obs"].to(torch.float32)))}

class PPORLModuleWithSharedGlobalEncoder(TorchActionMaskRLM, PPOTorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(
        self, config
    ) -> None:
        super().__init__(config=config)
        self.encoder = config.model_config_dict["shared_encoder"]
        config.model_config_dict.pop("shared_encoder")

class PPOModuleWithSharedEncoder(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        input_dim = module_spec.observation_space["observation"].shape[0]
        hidden_dims = module_spec.model_config_dict["fcnet_hiddens"]

        encoder_config = SharedEncoderConfig(input_dim=input_dim, hidden_dims=hidden_dims, freeze=False)
        # Since we want to use PPO, which is an actor-critic algorithm, we need to
        # use an ActorCriticEncoderConfig to wrap the base encoder config.
        actor_critic_encoder_config = ActorCriticEncoderConfig(
            base_encoder_config=encoder_config
        )

        shared_encoder = actor_critic_encoder_config.build(framework="torch")

        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            module_spec.update(SingleAgentRLModuleSpec(
                model_config_dict={
                    "shared_encoder": shared_encoder,
                } | module_spec.model_config_dict
            ))
            rl_modules[module_id] = module_spec.build()

        self._rl_modules = rl_modules

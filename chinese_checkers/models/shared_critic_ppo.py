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


class PPORLModuleWithSharedGlobalEncoder(TorchActionMaskRLM):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(
        self, encoder: nn.Module, local_dim: int, hidden_dim: int, action_dim: int
    ) -> None:
        super().__init__(config=None)
        self.encoder = encoder

    # def _forward_inference(self, batch):
    #     with torch.no_grad():
    #         return self._common_forward(batch)

    # def _forward_exploration(self, batch):
    #     with torch.no_grad():
    #         return self._common_forward(batch)

    # def _forward_train(self, batch):
    #     return self._common_forward(batch)

    # def _common_forward(self, batch):
    #     obs = batch["obs"]
    #     global_enc = self.encoder(obs["global"])
    #     policy_in = torch.cat([global_enc, obs["local"]], dim=-1)
    #     action_logits = self.policy_head(policy_in)

    #     return {"action_dist": torch.distributions.Categorical(logits=action_logits)}


class PPOModuleWithSharedEncoder(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):

        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        global_dim = module_spec.observation_space["global"].shape[0]
        hidden_dim = module_spec.model_config_dict["fcnet_hiddens"][0]
        shared_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            rl_modules[module_id] = PPORLModuleWithSharedGlobalEncoder(
                encoder=shared_encoder,
                local_dim=module_spec.observation_space.shape[0],
                hidden_dim=hidden_dim,
                action_dim=module_spec.action_space.n,
            )

        self._rl_modules = rl_modules

config = (
    PPOConfig()
    .rl_module(
        rl_module_spec=SingleAgentRLModuleSpec(module_class=MobileNetTorchPPORLModule)
    )
    .environment(
        RandomEnv,
        env_config={
            "action_space": gym.spaces.Discrete(2),
            # Test a simple Image observation space.
            "observation_space": gym.spaces.Box(
                0.0,
                1.0,
                shape=MOBILENET_INPUT_SHAPE,
                dtype=np.float32,
            ),
        },
    )
    # The following training settings make it so that a training iteration is very
    # quick. This is just for the sake of this example. PPO will not learn properly
    # with these settings!
    .training(train_batch_size=32, sgd_minibatch_size=16, num_sgd_iter=1)
)

config.build().train()
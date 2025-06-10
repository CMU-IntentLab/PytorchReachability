import argparse
import os
import sys
import pprint

import gymnasium #as gym
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(saferl_dir)
print(sys.path)
import models
import tools
import ruamel.yaml as yaml
import wandb
from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs

from termcolor import cprint
from datetime import datetime
import pathlib
from pathlib import Path
import collections
from PIL import Image
import io
from PyHJ.data import Batch
import matplotlib.pyplot as plt
# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from scripts.generate_data_traj_cont import get_frame
from load_config import load_config


def load_models():
    config = load_config("/content/PytorchReachability/configs.yaml")
    env = gymnasium.make(config.task, params = [config])
    config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)

    env = gymnasium.make(config.task, params = [config])

    # check if the environment has control and disturbance actions:
    assert hasattr(env, 'action_space') #and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
    config.state_shape = env.observation_space.shape or env.observation_space.n
    config.action_shape = env.action_space.shape or env.action_space.n
    config.max_action = env.action_space.high[0]

    if config.actor_activation == 'ReLU':
        actor_activation = torch.nn.ReLU
    elif config.actor_activation == 'Tanh':
        actor_activation = torch.nn.Tanh
    elif config.actor_activation == 'Sigmoid':
        actor_activation = torch.nn.Sigmoid
    elif config.actor_activation == 'SiLU':
        actor_activation = torch.nn.SiLU

    if config.critic_activation == 'ReLU':
        critic_activation = torch.nn.ReLU
    elif config.critic_activation == 'Tanh':
        critic_activation = torch.nn.Tanh
    elif config.critic_activation == 'Sigmoid':
        critic_activation = torch.nn.Sigmoid
    elif config.critic_activation == 'SiLU':
        critic_activation = torch.nn.SiLU

    if config.critic_net is not None:
        critic_net = Net(
            config.state_shape,
            config.action_shape,
            hidden_sizes=config.critic_net,
            activation=critic_activation,
            concat=True,
            device=config.device
        )
    else:
        # report error:
        raise ValueError("Please provide critic_net!")

    critic = Critic(critic_net, device=config.device).to(config.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay_pyhj)


    from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

    print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

    actor_net = Net(config.state_shape, hidden_sizes=config.control_net, activation=actor_activation, device=config.device)
    actor = Actor(
        actor_net, config.action_shape, max_action=config.max_action, device=config.device
    ).to(config.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)


    policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=config.tau,
    gamma=config.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=config.exploration_noise),
    reward_normalization=config.rew_norm,
    estimation_step=config.n_step,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=config.actor_gradient_steps,
    )


    return wm, policy

if __name__ == "__main__":
    wm, policy = load_models()
    print('loaded world model')
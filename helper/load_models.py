import gymnasium #as gym
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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


def load_models():
    env = gymnasium.make("dubins-wm-colab")
    num_actions =  env.action_space.shape[0]
    wm = models.WorldModel(env.observation_space_full, env.action_space, 0)

    env = gymnasium.make("dubins-wm-colab")

    # check if the environment has control and disturbance actions:
    assert hasattr(env, 'action_space') #and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    actor_activation = torch.nn.ReLU
    critic_activation = torch.nn.ReLU
    

    critic_net = Net(
        state_shape,
        action_shape,
        hidden_sizes=[128,128,128],
        activation=critic_activation,
        concat=True,
        device="cuda"
    )
    
    critic = Critic(critic_net, device='cuda').to('cuda')
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=0.001, weight_decay=0.001)


    from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

    print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

    actor_net = Net(state_shape, hidden_sizes=[128,128,128], activation=actor_activation, device="cuda")
    actor = Actor(
        actor_net, action_shape, max_action=max_action, device="cuda"
    ).to("cuda")
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=1e-4)


    policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=0.005,
    gamma=0.9999,
    exploration_noise=GaussianNoise(sigma=0.1),
    reward_normalization=False,
    estimation_step=1,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=1,
    )


    return wm, policy

if __name__ == "__main__":
    wm, policy = load_models()
    print('loaded world model')

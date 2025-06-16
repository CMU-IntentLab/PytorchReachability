import argparse
import os
import sys

import gymnasium 
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from dino_wm.dino_models import VideoTransformer
from dino_wm.test_loader import SplitTrajectoryDataset
import ruamel.yaml as yaml

from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import  WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic

from termcolor import cprint
from datetime import datetime
import pathlib
#from dreamer import make_dataset
# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
from torch.utils.data import DataLoader


wm = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=8,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=3,
        dropout=0.1
    )

wm.load_state_dict(torch.load('/home/kensuke/latent-test/PytorchReachability/dino_wm/checkpoints/best_classifier.pth'))

hdf5_file = '/data/ken/latent-unsafe/consolidated.h5'
bs = 1
bl=20
device = 'cuda:0'
H = 3
expert_data = SplitTrajectoryDataset(hdf5_file, 3, split='train', num_test=0)

expert_loader = iter(DataLoader(expert_data, batch_size=1, shuffle=True))

env = gymnasium.make('franka_wm_DINO-v0', params = [wm, expert_data], device=device)


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
max_action = env.action_space.high[0]
action_shape = env.action_space.shape or env.action_space.n
amax_action = env.action_space.high[0]


train_envs = DummyVectorEnv(
    [lambda: gymnasium.make('franka_wm_DINO-v0', params = [wm, expert_data]) for _ in range(1)]
)
test_envs = DummyVectorEnv(
    [lambda: gymnasium.make('franka_wm_DINO-v0', params = [wm, expert_data]) for _ in range(1)]
)


# seed
np.random.seed(0)
torch.manual_seed(0)
train_envs.seed(0)
test_envs.seed(0)
# model

actor_activation = torch.nn.ReLU
critic_activation = torch.nn.ReLU


critic_net = Net(
    state_shape,
    action_shape,
    hidden_sizes=[512, 512, 512,512],
    activation=critic_activation,
    concat=True,
    device=device
)

critic = Critic(critic_net, device=critic_net.device).to(critic_net.device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-3)


from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")


actor_net = Net(state_shape, hidden_sizes=[512, 512, 512, 512], activation=actor_activation, device=device)
actor = Actor(
    actor_net, action_shape, max_action=max_action, device=device
).to(device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

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

log_path = os.path.join("logs/dinowm/")


# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(40000, len(train_envs)),
    exploration_noise=True
)
test_collector = Collector(policy, test_envs)


epoch = 0


def save_best_fn(policy, epoch=epoch):
    torch.save(
        policy.state_dict(), 
        os.path.join(
            log_path+"/epoch_id_{}".format(epoch),
            "rotvec_policy.pth"
        )
    )


def stop_fn(mean_rewards):
    return False

if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path+"/epoch_id_{}".format(epoch))


warmup = 1
total_eps = 15
for iter in range(warmup+total_eps):
    if iter  < warmup:
        policy._gamma = 0 # for warmup the value fn
        policy.warmup = True
    else:
        policy._gamma = 0.9999
        policy.warmup = False
    
    print("episodes: {}, remaining episodes: {}".format(iter, warmup+total_eps - iter))
    epoch = epoch + 1
    print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    if total_eps > 1:
        writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch))
    else:
        if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
            print("Just created the log directory!")
            print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
            os.makedirs(log_path+"/total_epochs_{}".format(epoch))
        writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch)) 
    
    logger = WandbLogger()
    logger.load(writer)
    
    # import pdb; pdb.set_trace()
    result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    1,
    40000, # steps per epoch
    8, # step per collect
    1, # test num
    512, # batch size
    update_per_step=0.125,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    logger=logger
    )
    
    save_best_fn(policy, epoch=epoch)
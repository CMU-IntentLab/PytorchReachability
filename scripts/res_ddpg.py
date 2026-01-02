import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic, ResCritic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs
# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
from PyHJ.data import Batch
from PyHJ.configs.ddpg import Args
import tyro
from tyro.conf import subcommand

from PyHJ.configs.ddpg import Args as DDPGArgs
import dataclasses
import typing
from PyHJ.utils.eval_utils import evaluate_V, evaluate_Q, find_a
from termcolor import cprint

def get_V(state, policy, critic, l_fn, args):
    if args.residual:
        val = evaluate_V(state, policy, critic)[0] + l_fn(state)
    else:
        val = evaluate_V(state, policy, critic)[0]

    # V(s) = min(l(s), max_a V(f(s,a)))
    val = np.minimum(val, l_fn(state))
    return val

def get_Q(state, action, critic, l_fn, args):
    if args.residual:
        val = evaluate_Q(state, action, critic)[0] + l_fn(state)
    else:
        val = evaluate_Q(state, action, critic)[0]

    return val


def make_critic(args):
    if args.critic_activation == 'ReLU':
        critic_activation = torch.nn.ReLU
    elif args.critic_activation == 'Tanh':
        critic_activation = torch.nn.Tanh
    elif args.critic_activation == 'Sigmoid':
        critic_activation = torch.nn.Sigmoid
    elif args.critic_activation == 'SiLU':
        critic_activation = torch.nn.SiLU

    critic_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.critic_net,
        activation=critic_activation,
        concat=True,
        device=args.device
    )
    if args.residual:
        critic = ResCritic(critic_net, device=args.device).to(args.device)
    else:
        critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    return critic, critic_optim

def make_actor(args):
    if args.actor_activation == 'ReLU':
        actor_activation = torch.nn.ReLU
    elif args.actor_activation == 'Tanh':
        actor_activation = torch.nn.Tanh
    elif args.actor_activation == 'Sigmoid':
        actor_activation = torch.nn.Sigmoid
    elif args.actor_activation == 'SiLU':
        actor_activation = torch.nn.SiLU

    actor_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.control_net,
        activation=actor_activation,
        device=args.device
    )
    actor = Actor(actor_net, args.action_shape, max_action=args.max_action, device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    return actor, actor_optim

def main(args):
    task = args.task
    print(f"task: {task}")
    if 'res' in task and not args.residual:
        print("Residual critic is required for this task")
        exit()
    if 'res' not in task and args.residual:
        print("Residual critic is not required for this task")
        exit()
    env = gym.make(args.task)
    assert hasattr(env, 'action_space')
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    args.action1_shape = env.action_space.shape or env.action_space.n
    args.max_action1 = env.action_space.high[0]


    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model

    actor, actor_optim = make_actor(args)
    critic, critic_optim = make_critic(args)

    log_path = None

    if args.residual:
        from PyHJ.policy import avoid_DDPGPolicy_annealing_residual as DDPGPolicy
        print("Using residual critic")
    else:
        from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
        print("Using non-residual critic")
    l_fn = env.l_fn

    print("DDPG under the avoid-RL Bellman equation has been loaded!")


    policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=args.actor_gradient_steps,
    l_fn=l_fn,
    )

    log_path = os.path.join(args.logdir, args.task, 'ddpg_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_gamma_{}'.format(
    args.actor_activation, 
    args.critic_activation, 
    args.actor_gradient_steps,args.tau, 
    args.training_num, 
    args.buffer_size,
    args.critic_net[0],
    len(args.critic_net),
    args.control_net[0],
    len(args.control_net),
    args.gamma
    ))



    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)



    epoch = 0
    log_path = log_path+'/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
            args.exploration_noise, 
            args.actor_lr, 
            args.critic_lr, 
            args.batch_size,
            args.step_per_epoch,
            args.kwargs,
            args.seed
        )


    def save_best_fn(policy, epoch=epoch):
        torch.save(
            policy.state_dict(), 
            os.path.join(
                log_path+"/epoch_id_{}".format(epoch),
                "policy.pth"
            )
        )


    def stop_fn(mean_rewards):
        return False


    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
        print("Just created the log directory!")
        # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))

    if not os.path.exists(log_path+"/fig"):
            print("Just created the fig directory!")
            os.makedirs(log_path+"/fig")
    for iter in range(args.total_episodes):
        if iter < args.warmup_epoch:
            policy._gamma = 0.
            policy.warmup = True
            print("Warmup mode")
        else:
            policy._gamma = args.gamma
            policy.warmup = False
            print("Training mode")
        if args.continue_training_epoch is not None:
            print("episodes: {}, remaining episodes: {}".format(epoch//args.epoch, args.total_episodes - iter))
        else:
            print("episodes: {}, remaining episodes: {}".format(iter, args.total_episodes - iter))
        
        epoch = epoch + args.epoch
        if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
            print("Just created the log directory!")
            # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
            os.makedirs(log_path+"/epoch_id_{}".format(epoch))
        print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        if args.total_episodes > 1:
            writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        else:
            if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
                print("Just created the log directory!")
                print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
                os.makedirs(log_path+"/total_epochs_{}".format(epoch))
            writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        
        logger = TensorboardLogger(writer)
        # import pdb; pdb.set_trace()
        result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
        )
        save_best_fn(policy, epoch=epoch)

        #fig1, fig2 = env.get_eval_plot(policy, critic)
        #fig1.savefig(log_path+f"/fig/eval_plot_{iter}_bin.png")
        #fig2.savefig(log_path+f"/fig/eval_plot_{iter}_f2.png")
        policy.eval()
        deltas, vals, real_vals = rollout_eval(policy, critic, env, l_fn, args, num_rollouts=50)
        policy.train()
        cprint(f"Avg Overestimation: {np.mean(deltas):.4f} +/- {np.std(deltas):.4f}", "red")
        cprint(f"Avg Error: {np.mean(np.abs(deltas)):.4f} +/- {np.std(np.abs(deltas)):.4f}", "red")

    #make_gif_from_eval_plots(log_path, pattern="fig/eval_plot_*_f2.png", gif_name="eval_plots.gif", duration=500)
    #make_gif_from_eval_plots(log_path, pattern="fig/eval_plot_*_bin.png", gif_name="eval_plots_binary.gif", duration=500)
    

def rollout_eval(policy, critic, env, l_fn,args, num_rollouts=10):
    deltas = []
    vals = []
    real_vals = []
    for i in range(num_rollouts):
        state, _ = env.reset()
        done = False
        if args.off_policy:
            init_ac = env.action_space.sample()
        else:
            init_ac = find_a(state, policy)
        val = get_Q(state, init_ac, critic, l_fn, args)
        vals.append(val)
        state, reward, terminated, truncated, info = env.step(init_ac)
        rews = [l_fn(state)]
        # predicted value
        while not done:
            ac = find_a(state, policy)
            state, reward, terminated, truncated, info = env.step(ac)
            rew = l_fn(state)
            done = terminated or truncated
            rews.append(rew)
        real_val = np.min(rews)
        real_vals.append(real_val)
        delta = val - real_val
        deltas.append(delta)
    return deltas, vals, real_vals

import glob
from PIL import Image

def make_gif_from_eval_plots(log_path, pattern="fig/eval_plot_*_f2.png", gif_name="eval_plots.gif", duration=500):
    # Find all matching files and sort by iteration order
    image_files = sorted(
        glob.glob(os.path.join(log_path, pattern)),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[2])
    )
    if not image_files:
        print(f"No files matching {pattern} found in {log_path}")
        return

    # Load images
    images = [Image.open(img_path) for img_path in image_files]
    # Save as GIF
    gif_path = os.path.join(log_path, gif_name)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    print(f"GIF saved to {gif_path}")


default_configs = {
    "debug": (
        "debug experiment.",
        DDPGArgs(
            total_episodes=1,
            step_per_epoch=10000,
            warmup_epoch=0
        ),
    ),

    "default": (
        "standard experiment.",
        DDPGArgs(),
    ),
}


if __name__ == "__main__":
    import tyro

    cfg = tyro.extras.overridable_config_cli(default_configs)
    main(cfg)

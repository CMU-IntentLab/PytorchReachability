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
from PyHJ.utils.net.continuous import ActorProb, Critic, ResCritic
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
        # evaluate_V already extracts residual component for ResCritic
        val = evaluate_V(state, policy, critic)[0] + l_fn(state)
    else:
        val = evaluate_V(state, policy, critic)[0]

    return val

def get_Q(state, action, critic, l_fn, args):
    # evaluate_Q already extracts residual component for ResCritic
    val = evaluate_Q(state, action, critic)
    pred_q = val[0]
    pred_l = val[1]
    if args.residual:
        print('l_fn', l_fn(state))
        print('pred_l', pred_l)
        print('res val', pred_q)
        print('res val + l_fn', pred_q + l_fn(state))
        pred_q += l_fn(state)
    return pred_q


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
        critic1 = ResCritic(critic_net, device=args.device).to(args.device)
        critic2 = ResCritic(critic_net, device=args.device).to(args.device)
    else:
        critic1 = Critic(critic_net, device=args.device).to(args.device)
        critic2 = Critic(critic_net, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    return critic1, critic1_optim, critic2, critic2_optim

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
    actor = ActorProb(actor_net, args.action_shape, device=args.device).to(args.device)
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
    critic1, critic1_optim, critic2, critic2_optim = make_critic(args)

    log_path = None

    if args.residual:
        from PyHJ.policy.modelfree.sac_avoid_classical_residual import avoid_SACPolicy_annealing_residual as SACPolicy
        print("Using residual critic")
    else:
        from PyHJ.policy import avoid_SACPolicy_annealing as SACPolicy
        print("Using non-residual critic")
    l_fn = env.l_fn

    print("SAC under the avoid-RL Bellman equation has been loaded!")

    # Setup alpha for entropy regularization
    if args.auto_alpha:
        target_entropy = -0.5 * np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    policy = SACPolicy(
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        alpha=alpha,
        exploration_noise=None,  # SAC doesn't use GaussianNoise like DDPG
        deterministic_eval=True,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=env.action_space,
        actor1=actor,
        actor1_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
        l_fn=l_fn,
    )

    log_path = os.path.join(args.logdir, args.task, 'sac_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_gamma_{}'.format(
    args.actor_activation, 
    args.critic_activation, 
    args.actor_gradient_steps,args.tau, 
    args.training_num, 
    args.buffer_size,
    args.critic_net[0],
    len(args.critic_net),
    args.control_net[0],
    len(args.control_net),
    args.gamma_pyhj
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
            args.batch_size_pyhj,
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
            policy._gamma = args.gamma_pyhj
            policy.warmup = False
            print("Training mode")
        if args.continue_training_epoch is not None:
            print("episodes: {}, remaining episodes: {}".format(epoch//args.epoch, args.total_episodes - iter))
        else:
            print("episodes: {}, remaining episodes: {}".format(iter, args.total_episodes - iter))
        
        epoch = epoch + args.epoch
        if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
            #print("Just created the log directory!")
            # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
            os.makedirs(log_path+"/epoch_id_{}".format(epoch))
        #print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        if args.total_episodes > 1:
            writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        else:
            if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
                #print("Just created the log directory!")
                #print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
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
        args.batch_size_pyhj,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
        )
        save_best_fn(policy, epoch=epoch)

        #fig1, fig2 = env.get_eval_plot(policy, critic1)
        #fig1.savefig(log_path+f"/fig/eval_plot_{iter}_bin.png")
        #fig2.savefig(log_path+f"/fig/eval_plot_{iter}_f2.png")
        policy.eval()
        deltas, vals, real_vals = rollout_eval(policy, critic1, env, l_fn, args, num_rollouts=50)
        policy.train()
        cprint(f"Avg Overestimation: {np.mean(deltas):.4f} +/- {np.std(deltas):.4f}", "red")
        cprint(f"Avg Error: {np.mean(np.abs(deltas)):.4f} +/- {np.std(np.abs(deltas)):.4f}", "red")
    #make_gif_from_eval_plots(log_path, pattern="fig/eval_plot_*_f2.png", gif_name="eval_plots.gif", duration=500)
    #make_gif_from_eval_plots(log_path, pattern="fig/eval_plot_*_bin.png", gif_name="eval_plots_binary.gif", duration=500)
    

def rollout_eval(policy, critic, env, l_fn,args, num_rollouts=10):
    if args.off_policy:
        cprint("Off-policy evaluation", "blue")
    else:
        cprint("On-policy evaluation", "blue")
    deltas = []
    vals = []
    real_vals = []
    hj_rews = []

    for i in range(num_rollouts):
        state, _ = env.reset()
        rews = [l_fn(state)]
        done = False
        if args.off_policy:
            init_ac = env.action_space.sample()
        else:
            init_ac = find_a(state, policy)
        cprint('**env reset!**', 'blue')
        val = get_Q(state, init_ac, critic, l_fn, args)
        vals.append(val)
        state, reward, terminated, truncated, info = env.step(init_ac)
        hj_rews.append(reward)
        # predicted value
        while not done:
            rew = l_fn(state)
            rews.append(rew)
            ac = find_a(state, policy)            
            state, reward, terminated, truncated, info = env.step(ac)
            hj_rews.append(reward)
            done = terminated or truncated
        rew = l_fn(state)
        rews.append(rew)
        real_val = np.min(rews)
        
        real_vals.append(real_val)
        delta = val - real_val
        print('real_val', real_val)
        print('delta', delta)
        deltas.append(delta)
    print('hj_rews', np.mean(hj_rews), np.std(hj_rews), np.max(hj_rews), np.min(hj_rews))

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


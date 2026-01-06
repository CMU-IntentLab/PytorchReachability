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
from PyHJ.utils.net.continuous import Actor, ActorProb, Critic, ResCritic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs
# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
from PyHJ.data import Batch
import tyro
from tyro.conf import subcommand

from PyHJ.configs.train_conf import Args
import dataclasses
from typing import Callable, Optional, Tuple
from PyHJ.utils.eval_utils import evaluate_V, evaluate_Q, find_a
from termcolor import cprint
import wandb
from PyHJ.utils.logger.base import BaseLogger, LOG_DATA_TYPE

class WandbOnlyLogger(BaseLogger):
    """A logger that only uses wandb, without tensorboard dependency."""
    
    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.save_interval = 1
        self.last_save_step = -1
    
    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        """Log data directly to wandb."""
        # Convert numpy types to Python native types for wandb
        log_data = {}
        for k, v in data.items():
            if isinstance(v, (np.integer, np.floating)):
                log_data[k] = v.item()
            elif isinstance(v, np.ndarray):
                log_data[k] = v.tolist() if v.size > 1 else v.item()
            else:
                log_data[k] = v
        wandb.log(log_data, step=step)
    
    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        """Save checkpoint metadata to wandb."""
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)
            artifact = wandb.Artifact(
                f'checkpoint_epoch_{epoch}',
                type='model',
                metadata={
                    "save/epoch": epoch,
                    "save/env_step": env_step,
                    "save/gradient_step": gradient_step,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
    
    def restore_data(self) -> Tuple[int, int, int]:
        """Return default values (no restoration from wandb)."""
        return 0, 0, 0

activation_dict = {}
activation_dict['ReLU'] = torch.nn.ReLU
activation_dict['Tanh'] = torch.nn.Tanh
activation_dict['Sigmoid'] = torch.nn.Sigmoid
activation_dict['SiLU'] = torch.nn.SiLU

def get_V(state, policy, critic, l_fn, args):
    if args.residual:
        val = evaluate_V(state, policy, critic)[0] + l_fn(state)
    else:
        val = evaluate_V(state, policy, critic)[0]
    
    val = np.minimum(val, l_fn(state))
    return val

def get_Q(state, action, critic, l_fn, args):
    if args.residual:
        val = evaluate_Q(state, action, critic)[0] + l_fn(state)
    else:
        val = evaluate_Q(state, action, critic)[0]
    
    val = np.minimum(val, l_fn(state))
    return val

def make_critic(args):
    critic_activation = activation_dict[args.critic_activation]

    if args.mode == 'sac':
        # SAC needs 2 independent critics - create separate network instances
        critic_net1 = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.critic_net,
            activation=critic_activation,
            concat=True,
            device=args.device
        )
        critic_net2 = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.critic_net,
            activation=critic_activation,
            concat=True,
            device=args.device
        )
        if args.residual:
            critic1 = ResCritic(critic_net1, device=args.device).to(args.device)
            critic2 = ResCritic(critic_net2, device=args.device).to(args.device)
        else:
            critic1 = Critic(critic_net1, device=args.device).to(args.device)
            critic2 = Critic(critic_net2, device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
        return critic1, critic1_optim, critic2, critic2_optim
    else:
        # DDPG needs 1 critic
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
    actor_activation = activation_dict[args.actor_activation]
    actor_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.control_net,
        activation=actor_activation,
        device=args.device
    )
    if args.mode == 'sac':
        actor = ActorProb(actor_net, args.action_shape, device=args.device).to(args.device)
    else:
        actor = Actor(actor_net, args.action_shape, max_action=args.max_action, device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    return actor, actor_optim

def main(args):
    # Validate mode
    if args.mode not in ['sac', 'ddpg']:
        raise ValueError("--mode must be 'sac' or 'ddpg'")
    
    assert 'res' not in args.task, "deprecated"
    task = args.task
    if args.residual:
        task = 'res-'+task
    print(f"task: {task}")
    print(f"mode: {args.mode}")

    if args.use_wandb:
        # Initialize wandb - if running as sweep agent, config will be populated
        wandb.init(project=args.wandb_project, config={})
        
        # Generic override: apply any sweep parameters from wandb.config to args
        if wandb.config:
            wandb.run.name = task
            for key, value in wandb.config.items():
                if hasattr(args, key) and value is not None:
                    setattr(args, key, value)
                    print(f"Sweep override: {key} = {value}")
                    wandb.run.name += f"-{key}={value}"
    
    
    env = gym.make(task)
    assert hasattr(env, 'action_space')
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    train_envs = DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(args.test_num)]
    )
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # model creation
    actor, actor_optim = make_actor(args)
    if args.mode == 'sac':
        critic1, critic1_optim, critic2, critic2_optim = make_critic(args)
    else:
        critic, critic_optim = make_critic(args)

    l_fn = env.l_fn

    # Policy imports and setup
    if args.mode == 'sac':
        if args.residual:
            from PyHJ.policy.modelfree.sac_avoid_classical_residual import avoid_SACPolicy_annealing_residual as PolicyClass
            print("Using SAC with residual critic")
        else:
            from PyHJ.policy import avoid_SACPolicy_annealing as PolicyClass
            print("Using SAC with non-residual critic")
        print("SAC under the avoid-RL Bellman equation has been loaded!")
        
        # Setup alpha for entropy regularization
        if args.auto_alpha:
            target_entropy = -np.prod(env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = args.alpha
        
        policy = PolicyClass(
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
        mode_prefix = 'sac_avoid'
        eval_critic = critic1
    else:
        if args.residual:
            from PyHJ.policy import avoid_DDPGPolicy_annealing_residual as PolicyClass
            print("Using DDPG with residual critic")
        else:
            from PyHJ.policy import avoid_DDPGPolicy_annealing as PolicyClass
            print("Using DDPG with non-residual critic")
        print("DDPG under the avoid-RL Bellman equation has been loaded!")
        
        policy = PolicyClass(
            critic,
            critic_optim,
            tau=args.tau,
            gamma=args.gamma_pyhj,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            reward_normalization=args.rew_norm,
            estimation_step=args.n_step,
            action_space=env.action_space,
            actor=actor,
            actor_optim=actor_optim,
            actor_gradient_steps=args.actor_gradient_steps,
            l_fn=l_fn,
        )
        mode_prefix = 'ddpg_avoid'
        eval_critic = critic

    # Log path setup
    log_path = os.path.join(args.logdir, task, '{}_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_gamma_{}'.format(
        mode_prefix,
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,
        args.tau, 
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
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))

    if not os.path.exists(log_path+"/fig"):
        print("Just created the fig directory!")
        os.makedirs(log_path+"/fig")
    
    # Initialize wandb once before the loop (if using wandb and not already initialized)
    if args.use_wandb and wandb.run is None:
        args.wandb_name = f"{args.mode}_warmup_{args.warmup}_episodes_{args.total_episodes}_buffer_size_{args.buffer_size}_gamma_{args.gamma_pyhj}"
        # Convert dataclass to dict for wandb config logging
        config_dict = dataclasses.asdict(args)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config_dict
        )
    elif args.use_wandb and wandb.run is not None:
        # Update run name if not set by sweep
        if not wandb.run.name:
            args.wandb_name = f"{args.mode}_warmup_{args.warmup}_episodes_{args.total_episodes}_buffer_size_{args.buffer_size}_gamma_{args.gamma_pyhj}"
            wandb.run.name = args.wandb_name
        # Update config with current args (allows tracking non-swept params)
        wandb.config.update(dataclasses.asdict(args), allow_val_change=True)
    for iter in range(args.total_episodes):
        if iter == 0 and args.warmup:
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
            os.makedirs(log_path+"/epoch_id_{}".format(epoch))
        
        # Logger setup - use wandb-only logger or tensorboard
        #if args.use_wandb:
        logger = WandbOnlyLogger()
        #else:
        if args.total_episodes > 1:
            writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch))
        else:
            if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
                os.makedirs(log_path+"/total_epochs_{}".format(epoch))
            writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch))
        tb_logger = TensorboardLogger(writer)
        
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
            logger=tb_logger
        )
        save_best_fn(policy, epoch=epoch)

        policy.eval()
        is_off_policy = False
        deltas, vals, real_vals = rollout_eval(policy, eval_critic, env, l_fn, is_off_policy, args, num_rollouts=50)
        cprint(f"Avg Overestimation: {np.mean(deltas):.4f} +/- {np.std(deltas):.4f}", "red")
        cprint(f"Avg Error: {np.mean(np.abs(deltas)):.4f} +/- {np.std(np.abs(deltas)):.4f}", "red")
        
        

        is_off_policy = True
        off_policy_deltas, off_policy_vals, off_policy_real_vals = rollout_eval(policy, eval_critic, env, l_fn, is_off_policy, args, num_rollouts=50)
        cprint(f"Avg Overestimation: {np.mean(off_policy_deltas):.4f} +/- {np.std(off_policy_deltas):.4f}", "red")
        cprint(f"Avg Error: {np.mean(np.abs(off_policy_deltas)):.4f} +/- {np.std(np.abs(off_policy_deltas)):.4f}", "red")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "eval/on_policy_avg_overestimation": np.mean(deltas),
                "eval/on_policy_avg_error": np.mean(np.abs(deltas)),
                "eval/on_policy_avg_overestimation_std": np.std(deltas),
                "eval/on_policy_avg_error_std": np.std(np.abs(deltas)),
                "eval/off_policy_avg_overestimation": np.mean(off_policy_deltas),
                "eval/off_policy_avg_error": np.mean(np.abs(off_policy_deltas)),
                "eval/off_policy_avg_overestimation_std": np.std(off_policy_deltas),
                "eval/off_policy_avg_error_std": np.std(np.abs(off_policy_deltas)),
            })

        policy.train()
            

def rollout_eval(policy, critic, env, l_fn, off_policy, args, num_rollouts=10):
    if off_policy:
        cprint("Off-policy evaluation", "blue")
    else:
        cprint("On-policy evaluation", "blue")
    deltas = []
    vals = []
    real_vals = []
    hj_rews = [] if args.mode == 'sac' else None  # SAC tracks this, DDPG doesn't

    for i in range(num_rollouts):
        state, _ = env.reset()
        rews = [l_fn(state)]
        done = False
        if off_policy:
            init_ac = env.action_space.sample()
        else:
            init_ac = find_a(state, policy)
        
        val = get_Q(state, init_ac, critic, l_fn, args)
        vals.append(val)
        state, reward, terminated, truncated, info = env.step(init_ac)
        if args.mode == 'sac' and hj_rews is not None:
            hj_rews.append(reward)
        
        while not done:
            rew = l_fn(state)
            rews.append(rew)
            ac = find_a(state, policy)
            state, reward, terminated, truncated, info = env.step(ac)
            if args.mode == 'sac' and hj_rews is not None:
                hj_rews.append(reward)
            done = terminated or truncated
        rew = l_fn(state)
        rews.append(rew)
        real_val = np.min(rews)
        real_vals.append(real_val)
        delta = val - real_val
        
        deltas.append(delta)
    
    if args.mode == 'sac' and hj_rews is not None:
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





if __name__ == "__main__":
    import tyro

    cfg = tyro.cli(Args)
    main(cfg)

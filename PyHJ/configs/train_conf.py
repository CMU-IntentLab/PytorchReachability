from dataclasses import dataclass, field
from typing import Literal
from tyro.conf import FlagConversionOff
@dataclass
class Args:
  # Environment
  task: str = "dubins4d-v0"
  logdir: str = "logs"
  seed: int = 42
  device: str = "cuda" 
  reward_threshold: float | None = None 
  buffer_size: int = 100000 
  actor_lr: float = 1e-4
  critic_lr: float = 3e-4
  gamma_pyhj: float = 0.95 # type=float, default=0.95)
  tau: float = 0.005 # type=float, default=0.005)
  exploration_noise: float = 0.1 # type=float, default=0.1)
  epoch: int = 1 # type=int, default=10)
  total_episodes: int = 10 # type=int, default=160)
  step_per_epoch: int = 10000 # type=int, default=40000)
  step_per_collect: int = 8 # type=int, default=8)
  update_per_step: float = 0.125 # type=float, default=0.125)
  batch_size_pyhj: int = 512 # type=int, default=512)
  control_net: list[int] = field(default_factory=lambda: [128, 128, 128]) # type=int, nargs="*", default=None) # for control policy
  critic_net: list[int] = field(default_factory=lambda: [256, 256, 256])  # type=int, nargs="*", default=None) # for critic net
  training_num: int = 1 # type=int, default=8)
  test_num: int = 1 # type=int, default=100)
  render: float = 0. # type=float, default=0.)
  rew_norm: bool = False # action="store_true", default=False)
  n_step: int = 1 # type=int, default=1)
  continue_training_logdir: str | None = None # type=str, default=None)
  continue_training_epoch: int | None = None # type=int, default=None)
  actor_gradient_steps: int = 1 # type=int, default=1)
  target_update_freq: int = 400 # type=int, default=400)
  auto_alpha: int = 1
  alpha_lr: float = 3e-4
  alpha: float = 0.2 
  weight_decay_pyhj: float = 0.001
  actor_activation: str = "SiLU" #type=str, default="ReLU")
  critic_activation: str = "SiLU" # type=str, default="ReLU")
  kwargs: dict = field(default_factory=dict) # type=str, default="{}")


  residual: FlagConversionOff[bool] = False
  warmup: FlagConversionOff[bool] = False
  #off_policy: FlagConversionOff[bool] = False
  mode: Literal["sac", "ddpg"] = "ddpg"
  use_wandb: bool = True
  wandb_project: str | None = 'HJ-RL'

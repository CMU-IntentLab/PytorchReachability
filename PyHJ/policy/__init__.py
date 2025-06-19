"""Policy package."""
# isort:skip_file

from PyHJ.policy.base import BasePolicy
from PyHJ.policy.random import RandomPolicy
from PyHJ.policy.modelfree.ddpg import DDPGPolicy
from PyHJ.policy.modelfree.sac import SACPolicy


# the belows are new
from PyHJ.policy.modelfree.ddpg_reach_avoid_classical import reach_avoid_DDPGPolicy_annealing
from PyHJ.policy.modelfree.sac_reach_avoid_classical import reach_avoid_SACPolicy_annealing
from PyHJ.policy.modelfree.sac_avoid_classical import avoid_SACPolicy_annealing
from PyHJ.policy.modelfree.ddpg_avoid_classical import avoid_DDPGPolicy_annealing
from PyHJ.policy.modelfree.ddpg_avoid_classical_dinowm import avoid_DDPGPolicy_annealing_dinowm

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "DDPGPolicy",
    "SACPolicy",
    "reach_avoid_game_DDPGPolicy_annealing", # arXiv:2112.12288, implemented using DDPG
    "reach_avoid_game_SACPolicy_annealing", # arXiv:2112.12288, implemented using SAC
    "avoid_DDPGPolicy_annealing",
    "avoid_DDPGPolicy_annealing_dinowm",
    "avoid_SACPolicy_annealing"
]


"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)



# Customized environments begin: 


register(
    id='ra_highway_Game-v2',
    entry_point='PyHJ.reach_rl_gym_envs.ra_highway_10d:Highway_10D_game_Env2',
    max_episode_steps = 400, # horizon of the problem
    reward_threshold=1e8,
)
register(
    id="ra_droneracing_Game-v6",
    entry_point="PyHJ.reach_rl_gym_envs.Double_Drones_RA_linear:Double_Drones_RA_linear_Game_Env6",
    max_episode_steps=200,
    reward_threshold=1e8
)
register(
    id="ra_1d_Game-v0",
    entry_point="PyHJ.reach_rl_gym_envs.ra_1d:LQR_Env",
    max_episode_steps=1000,
    reward_threshold=1e8,
)

register(
    id="dubins-v0",
    entry_point="PyHJ.reach_rl_gym_envs.dubins:Dubins_Env",
    max_episode_steps=1000,
    reward_threshold=1e8,
)

register(
    id="dubins4d-v0",
    entry_point="PyHJ.reach_rl_gym_envs.dubins4d:Dubins_Env_4D",
    max_episode_steps=1000,
    reward_threshold=1e8,
)

register(
    id="dubinsRA-v0",
    entry_point="PyHJ.reach_rl_gym_envs.dubinsRA:DubinsRA_Env",
    max_episode_steps=1000,
    reward_threshold=1e8,
)

register(
    id="dubins-wm",
    entry_point="PyHJ.reach_rl_gym_envs.dubins-wm:Dubins_WM_Env",
    max_episode_steps=16,
    reward_threshold=1e8,
)

register(
    id="franka_wm_DINO-v0",
    entry_point="PyHJ.reach_rl_gym_envs.franka-DINOwm:Franka_DINOWM_Env",
    max_episode_steps=10,
    reward_threshold=1e8,
)
import numpy as np
from typing import Any, Dict, Optional, Union
from PyHJ.data import Batch

def find_a(state, policy):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy(tmp_batch, model = "actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    return act

def evaluate_V(state, policy, critic):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    critic_output = critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    # Extract residual component (first column) for ResCritic
    if critic_output.shape[1] == 2:
        tmp = critic_output[:, 0]
    else:
        tmp = critic_output
    return tmp.cpu().detach().numpy().flatten()

def evaluate_Q(state, action, critic):
    #print(state.shape, action.shape)
    tmp_obs = np.array(state).reshape(1,-1)
    action = np.array(action).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())

    tmp = critic(tmp_batch.obs, action)
    
    return tmp.cpu().detach().numpy().flatten()
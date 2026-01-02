from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from PyHJ.utils import evaluate_V



class Dubins_Env_4D(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.u_max = 1.25
        self.a_max = 1
        self.v_max = 3
        self.high = np.array([
            1.1, 1.1, 1.1, 1.1, self.v_max
        ])
        self.low = np.array([
            -1.1, -1.1, -1.1, -1.1, 0
        ])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # joint action space
        
        self.constraint = [0., 0., 0.5] # constraint: [x, y, r]
        
    def step(self, action):
        # l(x) = (x-x0)^2 + (y-y0)^2 - r^2
        rew = self.l_fn(self.state)

        # action is in -1, 1. Scale by self.u_max
        self.state[0] = self.state[0] + self.dt * self.state[3] * self.state[4]
        self.state[1] = self.state[1] + self.dt * self.state[2] * self.state[4]
        
        # this trick ensure the periodic state is continuous. this is important so the inputs the nn have no discontinuities
        theta = np.arctan2(self.state[2], self.state[3]) # sin, cos
        theta_next = theta + self.dt * (action[0] * self.u_max)
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        self.state[4] = self.state[4] + self.dt * (action[1] * self.a_max)
        self.state[4] = np.clip(self.state[4], 0, self.v_max)

        
        terminated = False
        truncated = False
        if any(self.state[:2] > self.high[:2]) or any(self.state[:2] < self.low[:2]):
            terminated = True
        
        info = {}
        return self.state.astype(np.float32), rew, terminated, truncated, info
    
    def l_fn(self, state):
        if state.ndim == 1:
            rew = (state[0]-self.constraint[0])**2 + (state[1]-self.constraint[1])**2 - self.constraint[2]**2
        else:   
            rew = (state[:, 0]-self.constraint[0])**2 + (state[:, 1]-self.constraint[1])**2 - self.constraint[2]**2
        return rew
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            theta = np.random.uniform(low=0, high=2*np.pi)
            self.state = np.random.uniform(low=[-1.1, -1.1, -1.1, -1.1, 0.], high=[1.1, 1.1, 1.1, 1.1, self.v_max], size=(5,))
            self.state[2] = np.sin(theta)
            self.state[3] = np.cos(theta)        
        else:
            self.state = initial_state     
        self.state = self.state.astype(np.float32)   
        return self.state, {}

class Res_Dubins_Env_4D(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.u_max = 1.25
        self.a_max = 1
        self.v_max = 3
        self.high = np.array([
            1.1, 1.1, 1.1, 1.1, self.v_max
        ])
        self.low = np.array([
            -1.1, -1.1, -1.1, -1.1, 0
        ])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # joint action space
        
        self.constraint = [0., 0., 0.5] # constraint: [x, y, r]
    def step(self, action):
        # action is in -1, 1. Scale by self.u_max
        rew_cur = self.l_fn(self.state)

        self.state[0] = self.state[0] + self.dt * self.state[3] * self.state[4]
        self.state[1] = self.state[1] + self.dt * self.state[2] * self.state[4]
        
        theta = np.arctan2(self.state[2], self.state[3]) # sin, cos
        theta_next = theta + self.dt * (action[0] * self.u_max)
        # this trick ensure the periodic state is continuous. this is important so the inputs the nn have no discontinuities
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        self.state[4] = self.state[4] + self.dt * (action[1] * self.a_max)
        self.state[4] = np.clip(self.state[4], 0, self.v_max)
        
        rew_next = self.l_fn(self.state)
        # l(x) = (x-x0)^2 + (y-y0)^2 - r^2
        rew = rew_next - rew_cur
        rew = np.minimum(rew, 0)
        terminated = False
        truncated = False
        if any(self.state[:2] > self.high[:2]) or any(self.state[:2] < self.low[:2]):
            terminated = True
        if self.state[-1] == 0:
            terminated = True

        info = {'rew_next': rew_next, 'rew_cur': rew_cur, 'rew': rew}
        return self.state.astype(np.float32), rew, terminated, truncated, info
    
    def l_fn(self, state):
        if state.ndim == 1:
            rew = (state[0]-self.constraint[0])**2 + (state[1]-self.constraint[1])**2 - self.constraint[2]**2
        else:   
            rew = (state[:, 0]-self.constraint[0])**2 + (state[:, 1]-self.constraint[1])**2 - self.constraint[2]**2
        return rew
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            theta = np.random.uniform(low=0, high=2*np.pi)
            self.state = np.random.uniform(low=[-1.1, -1.1, -1.1, -1.1, 0.], high=[1.1, 1.1, 1.1, 1.1, self.v_max], size=(5,))
            self.state[2] = np.sin(theta)
            self.state[3] = np.cos(theta)        
        else:
            self.state = initial_state     
        self.state = self.state.astype(np.float32)   
        return self.state, {}

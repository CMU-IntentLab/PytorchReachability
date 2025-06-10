from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.patches as patches
import torch
import math
class Dubins_WM_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self, params):
        
        if len(params) == 1:
            config = params[0]
        else:
            wm = params[0]
            past_data = params[1]
            config = params[2]
            self.set_wm(wm, past_data, config)


        self.render_mode = None
        self.time_step = 0.05
        self.high = np.array([
            1.1, 1.1, np.pi,
        ])
        self.low = np.array([
            -1.1, -1.1, -np.pi
        ])
        self.device = 'cuda:0'
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(544,), dtype=np.float32)
        image_size = config.size[0] #128
        img_obs_space = gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        obs_space = gym.spaces.Box(
                low=-1., high=1., shape=(2,), dtype=np.float32
            )
        bool_space = gym.spaces.Box(
                low=0., high=1., shape=(1,)
            )
        self.observation_space_full = gym.spaces.Dict({
            'image': img_obs_space,
            'obs_state': obs_space,
            'is_first': bool_space,
            'is_last': bool_space,
            'is_terminal': bool_space,
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # joint action space
        self.image_size=config.size[0]
        self.turnRate = config.turnRate


    def set_wm(self, wm, past_data, config):
        self.device = config.device
        self.encoder = wm.encoder.to(self.device)
        self.wm = wm.to(self.device)
        self.data = past_data
        
        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter
    
    def step(self, action):
        init = {k: v[:, -1] for k, v in self.latent.items()}
        ac_torch = torch.tensor([[action]], dtype=torch.float32).to(self.device)*self.turnRate
        self.latent = self.wm.dynamics.imagine_with_action(ac_torch, init)
        rew, cont = self.safety_margin(self.latent) # rew is negative if unsafe
        
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()
        if cont < 0.75:
            terminated = True
        else:
            terminated = False
        truncated = False
        info = {"is_first":False, "is_terminal":terminated}
        return np.copy(self.feat), rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
       
        
        init_traj = next(self.data)
        data = self.wm.preprocess(init_traj)
        embed = self.encoder(data)
        self.latent, _ = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        for k, v in self.latent.items(): 
            self.latent[k] = v[:, [-1]]
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy() 
        return np.copy(self.feat), {"is_first": True, "is_terminal": False}
      

    def safety_margin(self, state):
        g_xList = []
        
        feat = self.wm.dynamics.get_feat(state).detach()
        cont = self.wm.heads["cont"](feat)
        with torch.no_grad():  # Disable gradient calculation
            outputs = torch.tanh(self.wm.heads["margin"](feat))
            g_xList.append(outputs.detach().cpu().numpy())
        
        safety_margin = np.array(g_xList).squeeze()

        return safety_margin, cont.mean.squeeze().detach().cpu().numpy()
    
from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from PyHJ.utils import evaluate_V

class Dubins_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.high = np.array([
            1.1, 1.1, 1.1, 1.1
        ])
        self.low = np.array([
            -1.1, -1.1, -1.1, -1.1
        ])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # joint action space
        
        self.constraint = [0., 0., 0.5] # constraint: [x, y, r]
        self.u_max = 1.25

        
    def step(self, action):
        # action is in -1, 1. Scale by self.u_max

        self.state[0] = self.state[0] + self.dt * self.state[3]
        self.state[1] = self.state[1] + self.dt * self.state[2]
        
        theta = np.arctan2(self.state[2], self.state[3]) # sin, cos
        theta_next = theta + self.dt * (action[0] * self.u_max)


        # this trick ensure the periodic state is continuous. this is important so the inputs the nn have no discontinuities
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        # l(x) = (x-x0)^2 + (y-y0)^2 - r^2
        rew = ((self.state[0]-self.constraint[0])**2 + (self.state[1]-self.constraint[1])**2 - self.constraint[2]**2)
        
        terminated = False
        truncated = False
        if any(self.state[:2] > self.high[:2]) or any(self.state[:2] < self.low[:2]):
            terminated = True
        
        info = {}
        return self.state.astype(np.float32), rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            theta = np.random.uniform(low=0, high=2*np.pi)

            self.state = np.random.uniform(low=[-1, -1, -1, -1], high=[1, 1, 1, 1], size=(4,))
            self.state[2] = np.sin(theta)
            self.state[3] = np.cos(theta)        
        else:
            self.state = initial_state     
        self.state = self.state.astype(np.float32)   
        return self.state, {}


    def get_eval_plot(self, policy, critic):
        nx, ny = 51, 51
        thetas = [0, np.pi/6, np.pi/3, np.pi/2]

        fig1, axes1 = plt.subplots(1,len(thetas))
        fig2, axes2 = plt.subplots(1,len(thetas))
        X, Y = np.meshgrid(
            np.linspace(-1., 1., nx, endpoint=True),
            np.linspace(-1., 1., ny, endpoint=True),
        )
        for i in range(len(thetas)):
            V = np.zeros_like(X)
            for ii in range(nx):
                for jj in range(ny):
                    tmp_point = torch.tensor([
                                X[ii,jj],
                                Y[ii,jj],
                                np.sin(thetas[i]),
                                np.cos(thetas[i]),
                            ])
                    V[ii,jj] = evaluate_V( tmp_point, policy, critic)
            
        
            axes1[i].imshow(V>0, extent=(-1., 1., -1., 1.), origin='lower')
            axes2[i].imshow(V, extent=(-1., 1., -1., 1.), vmin=-1., vmax=1., origin='lower') 
            const = Circle((-0.0, -0.0), 0.5, color='red', fill=False)  # fill=False makes it a ring
            # Add the circle to the axes
            axes1[i].add_patch(const)
            const2 = Circle((-0.0, -0.0), 0.5, color='red', fill=False)  # fill=False makes it a ring
            # Add the circle to the axes
            axes2[i].add_patch(const2)   
            axes1[i].set_title('theta = {}'.format(np.round(thetas[i],2)), fontsize=12,)
            axes2[i].set_title('theta = {}'.format(np.round(thetas[i],2)), fontsize=12,)
            
                
        return fig1, fig2


    def get_eval_plot_f1(self, policy, critic):
        grid = np.load('/home/kensuke/HJRL/new_BRT_v1_w1.25.npy')

        plot_idxs = [0, 7, 14, 20]

        fig1, axes1 = plt.subplots(1,len(plot_idxs))
        fig2, axes2 = plt.subplots(1,len(plot_idxs))
        X, Y = np.meshgrid(
            np.linspace(-1.1, 1.1, grid.shape[0], endpoint=True),
            np.linspace(-1.1, 1.1, grid.shape[1], endpoint=True),
        )
        thetas_lin = np.linspace(0, 2*np.pi, grid.shape[2], endpoint=True)

        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(thetas_lin)):
            V = np.zeros_like(X)
            for ii in range(grid.shape[0]):
                for jj in range(grid.shape[1]):
                    tmp_point = torch.tensor([
                                X[ii,jj],
                                Y[ii,jj],
                                np.sin(thetas_lin[i]),
                                np.cos(thetas_lin[i]),
                            ])
                    V[ii,jj] = evaluate_V( tmp_point, policy, critic)
            
            V_grid = grid[:, :, i]
            tp_grid = np.sum((V>0) & (V_grid.T>0))
            fp_grid = np.sum((V>0) & (V_grid.T<0)) 
            fn_grid = np.sum((V<0) & (V_grid.T>0))
            tn_grid = np.sum((V<0) & (V_grid.T<0))
            tp += tp_grid
            fp += fp_grid
            fn += fn_grid
            tn += tn_grid

            prec_grid = tp_grid / (tp_grid + fp_grid) if (tp_grid + fp_grid) > 0 else 0
            rec_grid = tp_grid / (tp_grid + fn_grid) if (tp_grid + fn_grid) > 0 else 0
            f1_grid = 2 * (prec_grid * rec_grid) / (prec_grid + rec_grid) if (prec_grid + rec_grid) > 0 else 0
            
            if i in plot_idxs:
                plot_idx = plot_idxs.index(i)
                axes1[plot_idx].imshow(V>0, extent=(-1.1, 1.1, -1.1, 1.1), origin='lower')
                axes2[plot_idx].imshow(V, extent=(-1.1, 1.1, -1.1, 1.1), vmin=-1., vmax=1., origin='lower')    
                axes1[plot_idx].set_title('theta = {}'.format(np.round(thetas_lin[i],2)), fontsize=12,)
                axes2[plot_idx].set_title('f1 = {}'.format(np.round(f1_grid,2)), fontsize=12,)
                
                axes1[plot_idx].contour(grid[:,:,i].T, levels=[0.], colors='purple', linewidths=2, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
                
        print("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec =  tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec+ rec) if (prec + rec) > 0 else 0
        
        return fig1, fig2, f1



# Reachability in Pytorch

This is a minimal repository for doing HJ reachability analysis using the Discounted Safety/Reach-avoid Bellman equation originally introduced in [this](https://ieeexplore.ieee.org/abstract/document/8794107) and [this](https://arxiv.org/abs/2112.12288) paper. We build on the implementation used as baselines from [this repo](https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning). 


This repository supports SAC and DDPG implementations of both the safety-only and reach-avoid value functions. I have not yet tested HJ with disturbances.


We recommend Python version 3.12. 

Install instruction:

1. git clone the repo

2. cd to the root location of this repo, where you should be able to see the "setup.py". Note that if you use MacOS, then pytorch 2.4.0 is not available, and therefore you have to first change the line 22 of setup.py from "pytorch==2.4.0" to "pytorch==2.2.2", and then do the step 3. (However, Pytorch==2.4.0 is available for Ubuntu systems. So, if you use Ubuntu, then you can directly go to step 3. )

3. run in terminal: pip install -e .

4. run in terminal: conda install -c conda-forge ffmpeg


# Some sample training scripts:

For a Dubins Car Reach-avoid example: 

> python run_training_sac_RA_nodist.py --control-net 512 512 512 --critic-net 512 512 512 --epoch 1 --total-episodes 40

For a Dubins car avoid-only example: 

> python run_training_sac_nodist.py --control-net 512 512 512 --critic-net 512 512 512 --epoch 1 --total-episodes 40

Finally, we recommend always setting the action space to range from -1 to 1 in the gym.env definition, but we can scale or shift the actions within the gym.step() function when defining the dynamics. For example, if we have two double integrator dynamics: the first integrator’s control is bounded by -0.1 to 0.1, and the second integrator’s control is bounded by -0.3 to 0.3. In this case, we can define self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64) and implement the dynamics in gym.step(self, u) as follows:


**A key thing to notice is that the initial state distribution should cover a portion of the target set for reach-avoid settings. Otherwise, no state in the target set shows up in the data buffer and therefore, the policy cannot see where it should go to maximize the value function. **

In addition, we remark that the convergence of critic loss implies that the neural network value function approximates well the value function induced by the current learned policy. However, it does not mean the learning is done because we cannot tell the quality of policies by just looking at the critic loss. In minimax DDPG, it improves the learned policy by minimizing the control actor loss, and refines the disturbance policy by maximizing the disturbance actor loss. However, we observe that a small critic loss stabilizes the multi-agent reinforcement learning training, and therefore helps policy learning. 



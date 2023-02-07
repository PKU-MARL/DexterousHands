import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from bidexhands.algorithms.offrl.td3_bc import TD3_BC_Model
from bidexhands.algorithms.offrl.td3_bc import ReplayBuffer

class TD3_BC:

    def __init__(self,
                vec_env,
                device='cpu',
                discount = 0.99,
                tau = 0.005,
                alpha = 2.5,
                policy_freq = 2,
                batch_size = 250,
                max_timesteps = 1000000,
                iterations =  10000,
                log_dir = '',
                datatype = 'expert',
                algo = 'td3_bc'):

        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.vec_env = vec_env
        self.device = device
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps
        self.iterations = iterations
        self.log_dir = log_dir
        self.datatype = datatype
        self.algo = algo
        self.log_dir = self.log_dir.split(self.algo)[0]+self.algo+'/'+self.datatype+'/'
        self.data_dir = self.log_dir.split(self.algo)[0].split('logs')
        self.data_dir = self.data_dir[0]+'data'+self.data_dir[1]+self.datatype+'/'
        self.test_step = 40000

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir) 
        time.sleep(np.random.rand()*2)
        order = len(os.listdir(self.log_dir))
        self.reward_log = open(self.log_dir+str(order)+'.log','w')
        

    def run(self, num_learning_iterations, log_interval=1):
        
        current_obs = self.vec_env.reset()
        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0] 
        max_action = float(self.action_space.high[0])
        policy_noise = 0.2 * max_action
        noise_clip = 0.5 * max_action
    
        policy = TD3_BC_Model(state_dim, action_dim, max_action, self.device, self.discount, self.tau, policy_noise, noise_clip, self.policy_freq, self.alpha)

        replay_buffer = ReplayBuffer(state_dim, action_dim,self.device)
        replay_buffer.convert(self.data_dir)
        
        for t in range(int(self.max_timesteps/self.iterations)+1):
            policy.train(replay_buffer, self.iterations, self.batch_size)

            reward_sum = []
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            current_obs = self.vec_env.reset()
            for _ in range(int(self.test_step/self.vec_env.num_envs)):
                actions = policy.select_action(current_obs)
                next_obs, rews, dones, infos = self.vec_env.step(actions)
                current_obs.copy_(next_obs)
                cur_reward_sum[:] += rews
                new_ids = (dones > 0).nonzero(as_tuple=False)
                reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0

            self.reward_log.write(str(sum(reward_sum)/len(reward_sum))+'\n')
            self.reward_log.flush()
                    


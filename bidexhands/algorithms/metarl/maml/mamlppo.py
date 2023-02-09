from curses import meta
from datetime import datetime
import os
import time
from tokenize import PseudoExtras

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algorithms.metarl.maml import RolloutStorage

class MAMLPPO:

    def __init__(self,
                 vec_env,
                 pseudo_actor_critic,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate
        self.learning_rate = learning_rate

        self.num_envs = vec_env.num_envs
        self.num_env_each_task = int(vec_env.num_envs / vec_env.num_tasks)
        # PPO components
        self.vec_env = vec_env

        self.actor_critic = []
        self.optimizers = []
        self.task_num = self.vec_env.task_num
        for i in range(self.task_num):
            pseudo_actor_critic[i].to(self.device)
            self.optimizers.append(optim.SGD(pseudo_actor_critic[i].parameters(), lr=learning_rate))
            self.actor_critic.append(pseudo_actor_critic[i])

        self.storage = RolloutStorage(self.num_envs, num_transitions_per_env, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset
        self.train_epoch = 0

        self.rewbuffer = deque(maxlen=100)
        self.test_rewbuffer = deque(maxlen=100)
        self.task_rewbuffer = []
        for i in range(len(self.vec_env.task_envs) - 1):
            self.task_rewbuffer.append(deque(maxlen=100))
        self.lenbuffer = deque(maxlen=100)

        self.cur_reward_sum = torch.zeros(self.num_env_each_task * (self.task_num - 1), dtype=torch.float, device=self.device)
        self.cur_test_reward_sum = torch.zeros(self.num_env_each_task * 1, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.reward_sum = []
        self.test_reward_sum = []
        self.episode_length = []
        self.task_reward_sum = []
        for i in range(len(self.vec_env.task_envs) - 1):
            self.task_reward_sum.append([])
        self.support_storage_list = []

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)        

    def sample_support_trajectory(self, support_set_size, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        # rewbuffer = self.reward_sum
        # test_rewbuffer = self.test_rewbuffer
        # task_rewbuffer = self.task_rewbuffer
        # lenbuffer = self.lenbuffer

        # cur_reward_sum = self.cur_reward_sum
        # cur_test_reward_sum = self.cur_test_reward_sum
        # cur_episode_length = self.cur_episode_length

        # reward_sum = self.reward_sum
        # test_reward_sum = self.test_reward_sum
        # episode_length = self.episode_length
        # task_reward_sum = self.task_reward_sum

        self.support_storage_list = []

        for it in range(support_set_size):
            start = time.time()
            ep_infos = []

            # Rollout
            for _ in range(self.num_transitions_per_env):
                if self.apply_reset:
                    current_obs = self.vec_env.reset()
                    current_states = self.vec_env.get_state()
                # Compute the action
                for i in range(self.task_num):
                    if i == 0:
                        actions, actions_log_prob, values, mu, sigma = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                 current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                    else:
                        tem_actions, tem_actions_log_prob, tem_values, tem_mu, tem_sigma = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                 current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                        actions = torch.cat((actions, tem_actions), dim=0)
                        actions_log_prob = torch.cat((actions_log_prob, tem_actions_log_prob), dim=0)
                        values = torch.cat((values, tem_values), dim=0)
                        mu = torch.cat((mu, tem_mu), dim=0)
                        sigma = torch.cat((sigma, tem_sigma), dim=0)

                # Step the vec_environment
                next_obs, rews, dones, infos = self.vec_env.step(actions)
                next_states = self.vec_env.get_state()
                # Record the transition
                self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                current_obs.copy_(next_obs)
                current_states.copy_(next_states)
                # Book keeping
                ep_infos.append(infos)

                if self.print_log:
                    self.cur_reward_sum[:] += rews[:self.num_env_each_task * (self.task_num - 1)]
                    self.cur_test_reward_sum += rews[self.num_env_each_task * (self.task_num - 1):]
                    self.cur_episode_length[:] += 1

                    new_ids = (dones[:self.num_env_each_task * (self.task_num - 1)] > 0).nonzero(as_tuple=False)
                    test_new_ids = (dones[self.num_env_each_task * (self.task_num - 1):] > 0).nonzero(as_tuple=False)

                    for i in range(len(self.vec_env.task_envs) - 1):
                        task_new_ids = (dones[i*self.num_env_each_task:(i+1)*self.num_env_each_task] > 0).nonzero(as_tuple=False)
                        task_reward = self.cur_reward_sum[i*self.num_env_each_task:(i+1)*self.num_env_each_task]
                        self.task_reward_sum[i].extend(task_reward[task_new_ids][:, 0].cpu().numpy().tolist())

                    self.reward_sum.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.test_reward_sum.extend(self.cur_test_reward_sum[test_new_ids][:, 0].cpu().numpy().tolist())
                    self.episode_length.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_test_reward_sum[test_new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            if self.print_log:
                # reward_sum = [x[0] for x in reward_sum]
                # episode_length = [x[0] for x in episode_length]
                self.rewbuffer.extend(self.reward_sum)
                self.test_rewbuffer.extend(self.test_reward_sum)
                self.lenbuffer.extend(self.episode_length)
                for i in range(len(self.vec_env.task_envs) - 1):
                    self.task_rewbuffer[i].extend(self.task_reward_sum[i])

            for i in range(self.task_num):
                if i == 0:
                    _, _, last_values, _, _ = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                else:
                    _, _, tem_last_values, _, _ = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                    last_values = torch.cat((last_values, tem_last_values), dim=0)

            stop = time.time()
            collection_time = stop - start

            mean_trajectory_length, mean_reward = self.storage.get_statistics()

            # Learning step
            start = stop
            self.storage.compute_returns(last_values, self.gamma, self.lam)
            mean_value_loss, mean_surrogate_loss = self.update()
            self.train_epoch += 1

            self.support_storage_list.append(deepcopy(self.storage))
            self.storage.clear()
            stop = time.time()
            learn_time = stop - start
            if self.print_log:
                self.log(locals())
            ep_infos.clear()

        return self.support_storage_list

    def sample_query_trajectory(self, query_set_size, meta_storage):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        # rewbuffer = self.reward_sum
        # test_rewbuffer = self.test_rewbuffer
        # task_rewbuffer = self.task_rewbuffer
        # lenbuffer = self.lenbuffer

        # cur_reward_sum = self.cur_reward_sum
        # cur_test_reward_sum = self.cur_test_reward_sum
        # cur_episode_length = self.cur_episode_length

        # reward_sum = self.reward_sum
        # test_reward_sum = self.test_reward_sum
        # episode_length = self.episode_length
        # task_reward_sum = self.task_reward_sum
        self.query_storage_list = []

        for it in range(query_set_size):
            start = time.time()
            ep_infos = []

            # Rollout
            for _ in range(self.num_transitions_per_env):
                if self.apply_reset:
                    current_obs = self.vec_env.reset()
                    current_states = self.vec_env.get_state()
                # Compute the action
                for i in range(self.task_num):
                    if i == 0:
                        actions, actions_log_prob, values, mu, sigma = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                 current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                    else:
                        tem_actions, tem_actions_log_prob, tem_values, tem_mu, tem_sigma = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                 current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                        actions = torch.cat((actions, tem_actions), dim=0)
                        actions_log_prob = torch.cat((actions_log_prob, tem_actions_log_prob), dim=0)
                        values = torch.cat((values, tem_values), dim=0)
                        mu = torch.cat((mu, tem_mu), dim=0)
                        sigma = torch.cat((sigma, tem_sigma), dim=0)

                # Step the vec_environment
                next_obs, rews, dones, infos = self.vec_env.step(actions)
                next_states = self.vec_env.get_state()
                # Record the transition
                meta_storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                current_obs.copy_(next_obs)
                current_states.copy_(next_states)
                # Book keeping
                ep_infos.append(infos)

                if self.print_log:
                    self.cur_reward_sum[:] += rews[:self.num_env_each_task * (self.task_num - 1)]
                    self.cur_test_reward_sum += rews[self.num_env_each_task * (self.task_num - 1):]
                    self.cur_episode_length[:] += 1

                    new_ids = (dones[:self.num_env_each_task * (self.task_num - 1)] > 0).nonzero(as_tuple=False)
                    test_new_ids = (dones[self.num_env_each_task * (self.task_num - 1):] > 0).nonzero(as_tuple=False)

                    for i in range(len(self.vec_env.task_envs) - 1):
                        task_new_ids = (dones[i*self.num_env_each_task:(i+1)*self.num_env_each_task] > 0).nonzero(as_tuple=False)
                        task_reward = self.cur_reward_sum[i*self.num_env_each_task:(i+1)*self.num_env_each_task]
                        self.task_reward_sum[i].extend(task_reward[task_new_ids][:, 0].cpu().numpy().tolist())

                    self.reward_sum.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.test_reward_sum.extend(self.cur_test_reward_sum[test_new_ids][:, 0].cpu().numpy().tolist())
                    self.episode_length.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_test_reward_sum[test_new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            if self.print_log:
                # reward_sum = [x[0] for x in reward_sum]
                # episode_length = [x[0] for x in episode_length]
                self.rewbuffer.extend(self.reward_sum)
                self.test_rewbuffer.extend(self.test_reward_sum)
                self.lenbuffer.extend(self.episode_length)
                for i in range(len(self.vec_env.task_envs) - 1):
                    self.task_rewbuffer[i].extend(self.task_reward_sum[i])

            for i in range(self.task_num):
                if i == 0:
                    _, _, last_values, _, _ = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                else:
                    _, _, tem_last_values, _, _ = self.actor_critic[i].act(current_obs[self.num_env_each_task*i:self.num_env_each_task*(i+1)],
                                                                                                current_states[self.num_env_each_task*i:self.num_env_each_task*(i+1)])
                    last_values = torch.cat((last_values, tem_last_values), dim=0)


            stop = time.time()
            collection_time = stop - start

            mean_trajectory_length, mean_reward = meta_storage.get_statistics()
            self.train_epoch += 1

            # Learning step
            start = stop
            meta_storage.compute_returns(last_values, self.gamma, self.lam)
            stop = time.time()
            learn_time = stop - start
            self.query_storage_list.append(deepcopy(meta_storage))
            meta_storage.clear()
            if self.print_log:
                mean_value_loss, mean_surrogate_loss = 0, 0
                self.log(locals(), query=True)
            ep_infos.clear()

        return self.query_storage_list

    def log(self, locs, width=80, pad=35, query=False):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, self.train_epoch)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
            for i, task_name in enumerate(self.vec_env.task_envs):
                if i == len(self.vec_env.task_envs) - 1:
                    continue
                if len(self.task_rewbuffer[i]) > 0:
                    value = statistics.mean(self.task_rewbuffer[i])
                    self.writer.add_scalar('Episode/' + task_name, value, self.train_epoch)
                    ep_string += f"""{f'Mean Reward {task_name}:':>{pad}} {value:.4f}\n"""

        for i in range(self.task_num):
            if i != 0:
                mean_std += self.actor_critic[i].log_std.exp().mean()
            else:
                mean_std = self.actor_critic[i].log_std.exp().mean()
        fps = int(self.num_transitions_per_env * self.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], self.train_epoch)
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], self.train_epoch)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.train_epoch)
        if len(self.rewbuffer) > 0 and len(self.test_rewbuffer) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(self.rewbuffer), self.train_epoch)
            self.writer.add_scalar('Train/FPS',fps,self.train_epoch)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(self.lenbuffer), self.train_epoch)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(self.rewbuffer), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(self.lenbuffer), self.tot_time)
            self.writer.add_scalar('Train/mean_test_reward', statistics.mean(self.test_rewbuffer), self.train_epoch)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], self.train_epoch)
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], self.train_epoch)

        # fps = int(self.num_transitions_per_env * self.num_envs / (locs['collection_time'] + locs['learn_time']))
        str = f" \033[1m Inner update/Outer update: {self.train_epoch}/{self.train_epoch} \033[0m "

        if len(self.rewbuffer) > 0 and len(self.test_rewbuffer) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                          f"""{'Mean test reward:':>{pad}} {statistics.mean(self.test_rewbuffer):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.train_epoch + 1) * (
                               self.train_epoch - self.train_epoch):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
    
        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for i in range(self.task_num):
                obs_batch = self.storage.observations[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, *self.storage.observations.size()[2:])
                if self.asymmetric:
                    states_batch = self.storage.states[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, *self.storage.states.size()[2:])
                else:
                    states_batch = None
                actions_batch = self.storage.actions[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, self.storage.actions.size(-1))
                target_values_batch = self.storage.values[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                returns_batch = self.storage.returns[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                old_actions_log_prob_batch = self.storage.actions_log_prob[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                advantages_batch = self.storage.advantages[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                old_mu_batch = self.storage.mu[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, self.storage.actions.size(-1))
                old_sigma_batch = self.storage.sigma[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, self.storage.actions.size(-1))

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic[i].evaluate(obs_batch,
                                                                                            None,
                                                                                            actions_batch)

                # KL
                # if self.desired_kl != None and self.schedule == 'adaptive':
                #     kl = torch.sum(
                #         sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                #     kl_mean = torch.mean(kl)

                #     if kl_mean > self.desired_kl * 2.0:
                #         self.step_size = max(1e-5, self.step_size / 1.5)
                #     elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                #         self.step_size = min(1e-2, self.step_size * 1.5)

                #     for param_group in self.optimizers[i].param_groups:
                #         param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio

                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic[i].parameters(), self.max_grad_norm)
                self.optimizers[i].step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

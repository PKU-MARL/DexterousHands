from datetime import datetime
from operator import le
import os
import time
import random

from matplotlib.pyplot import step

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque
import copy
# from torchviz import make_dot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from bidexhands.algorithms.metarl.maml import RolloutStorage
from bidexhands.algorithms.metarl.maml.module import ActorCritic
import TorchOpt

class Trainer:

    def __init__(self,
                inner_algo,
                meta_actor_critic,
                vec_env,
                learning_rate=3e-4,
                sampler='sequential',
                asymmetric=False):

        self.num_mini_batches = inner_algo.num_mini_batches
        self.desired_kl = inner_algo.desired_kl
        self.schedule = inner_algo.schedule
        self.max_grad_norm = inner_algo.max_grad_norm
        self.clip_param = inner_algo.clip_param
        self.value_loss_coef = inner_algo.value_loss_coef
        self.entropy_coef = inner_algo.entropy_coef
        self.gamma = inner_algo.gamma
        self.lam = inner_algo.lam
        self.use_clipped_value_loss = inner_algo.use_clipped_value_loss
        self.step_size = inner_algo.step_size

        self.asymmetric = asymmetric
        self.sampler = sampler

        self.env = vec_env
        self.inner_algo = inner_algo

        self.task_num = self.env.task_num
        self.num_env_each_task = int(self.env.num_envs / self.env.num_tasks)
        self.support_set_size = 1
        self.query_set_size = 3

        self.num_meta_learning_epochs = 1
        self.meta_desired_kl = inner_algo.desired_kl
        self.meta_schedule = 'adaptive'

        self.meta_storage = RolloutStorage(self.env.num_envs, self.inner_algo.num_transitions_per_env, self.inner_algo.observation_space.shape,
                                      self.inner_algo.state_space.shape, self.inner_algo.action_space.shape, self.inner_algo.device, self.sampler)
        self.inner_algo = inner_algo
        self.meta_actor_critic = meta_actor_critic
        self.meta_actor_critic.to(self.inner_algo.device)
        # self.meta_optimizer = optim.Adam(self.meta_actor_critic.parameters(), lr=learning_rate)
        self.inner_optim = TorchOpt.MetaSGD(self.meta_actor_critic, lr=self.inner_algo.learning_rate)
        # self.inner_optim = optim.Adam(self.meta_actor_critic, lr=self.inner_algo.learning_rate)
        self.meta_optimizer = optim.Adam(self.meta_actor_critic.parameters(), lr=learning_rate)

    def train(self, train_epoch):
        for _ in range(train_epoch):
            support_storage_list = self.inner_algo.sample_support_trajectory(self.support_set_size)
            query_storage_list = self.inner_algo.sample_query_trajectory(self.query_set_size, self.meta_storage)

            self.meta_update(support_storage_list, query_storage_list)
            # clear the inner_algo parameters
            with torch.no_grad():
                for i in range(self.task_num):
                    for p_meta, p_inner in zip(self.meta_actor_critic.parameters(), self.inner_algo.actor_critic[i].parameters()):
                        polyak = 0
                        p_inner.data.mul_(polyak)
                        p_inner.data.add_((1 - polyak) * p_meta.data)

            if self.inner_algo.train_epoch == train_epoch:
                break

    def sample_batch_task(self, task_size):
        batch_task = random.sample(self.env.task_envs, task_size)
        return batch_task

    def meta_update(self, support_storage_list, query_storage_list):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        for query_storage in query_storage_list:
            outer_losses = []
            self.meta_optimizer.zero_grad()
            self.net_state = TorchOpt.extract_state_dict(self.meta_actor_critic, enable_visual=True, visual_prefix='step_0.') # extract state
            self.optim_state = TorchOpt.extract_state_dict(self.inner_optim)

            # -1 for test task
            for i in range(self.task_num - 1):
                for support_storage in support_storage_list:
                    self.inner_update(support_storage, i)

                obs_batch = query_storage.observations[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, *query_storage.observations.size()[2:])
                if self.asymmetric:
                    states_batch = query_storage.states[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, *query_storage.states.size()[2:])
                else:
                    states_batch = None
                actions_batch = query_storage.actions[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, query_storage.actions.size(-1))
                target_values_batch = query_storage.values[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                returns_batch = query_storage.returns[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                old_actions_log_prob_batch = query_storage.actions_log_prob[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                advantages_batch = query_storage.advantages[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
                old_mu_batch = query_storage.mu[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, query_storage.actions.size(-1))
                old_sigma_batch = query_storage.sigma[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, query_storage.actions.size(-1))

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.meta_actor_critic.evaluate(obs_batch,
                                                                                            None,
                                                                                            actions_batch)

                # KL
                # if self.meta_desired_kl != None and self.meta_schedule == 'adaptive':

                #     kl = torch.sum(
                #         sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                #     kl_mean = torch.mean(kl)

                #     if kl_mean > self.desired_kl * 2.0:
                #         self.step_size = max(1e-5, self.step_size / 1.5)
                #     elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                #         self.step_size = min(1e-2, self.step_size * 1.5)

                #     for param_group in self.meta_optimizer.param_groups:
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

                outer_losses.append(loss)
                outer_loss = (loss) / self.task_num
                outer_loss.backward()
                # recover network and optimizer states at the inital point for the next task
                TorchOpt.recover_state_dict(self.inner_optim, self.optim_state)
                TorchOpt.recover_state_dict(self.meta_actor_critic, self.net_state)

            outer_loss = sum(outer_losses) / self.task_num
            # Gradient step
            # make_dot(outer_loss, params=dict(self.meta_actor_critic.named_parameters()), show_attrs=True, show_saved=True)
            nn.utils.clip_grad_norm_(self.meta_actor_critic.parameters(), self.max_grad_norm)
            # for p_meta in self.meta_actor_critic.parameters():
            #     print(p_meta)
            # for name, parms in self.meta_actor_critic.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            #     ' -->grad_value:',parms.grad)
            self.meta_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_meta_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def inner_update(self, support_storage, i):
        for epoch in range(self.inner_algo.num_learning_epochs):
            obs_batch = support_storage.observations[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, *support_storage.observations.size()[2:])
            if self.asymmetric:
                states_batch = support_storage.states[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, *support_storage.states.size()[2:])
            else:
                states_batch = None
            actions_batch = support_storage.actions[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, support_storage.actions.size(-1))
            target_values_batch = support_storage.values[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
            returns_batch = support_storage.returns[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
            old_actions_log_prob_batch = support_storage.actions_log_prob[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
            advantages_batch = support_storage.advantages[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, 1)
            old_mu_batch = support_storage.mu[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, support_storage.actions.size(-1))
            old_sigma_batch = support_storage.sigma[:, self.num_env_each_task*i:self.num_env_each_task*(i+1), :].reshape(-1, support_storage.actions.size(-1))

            actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.meta_actor_critic.evaluate(obs_batch,
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

                # for param_group in self.meta_optimizer.param_groups:
                #     param_group['lr'] = self.step_size

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
            # for name, parms in self.meta_actor_critic.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            #     ' -->grad_value:',parms.grad)
            nn.utils.clip_grad_norm_(self.meta_actor_critic.parameters(), self.max_grad_norm)
            self.inner_optim.step(loss)
            

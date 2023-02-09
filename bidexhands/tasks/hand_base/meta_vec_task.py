# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from collections import defaultdict
from gym import spaces

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import torch
import numpy as np
import random


# VecEnv Wrapper for RL training
class VecTask():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.max_episode_length = self.task.max_episode_length
        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        # meta-rl
        self.activate_task_index = None
        self.task_id = 0
        self.mode = "vanilla"
        self.task_envs = task.task_envs
        self.num_tasks = len(task.task_envs)
        self.task_num = self.num_tasks
        self._sample_strategy = self.uniform_random_strategy
        self.num_env_each_task = int(self.num_environments / self.num_tasks)

        if self.mode == 'vanilla':
            self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
            self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        elif self.mode == 'add-onehot':
            self.obs_space = spaces.Box(np.ones(self.num_obs + self.num_tasks) * -np.Inf, np.ones(self.num_obs + self.num_tasks) * np.Inf)
            self.state_space = spaces.Box(np.ones(self.num_states + self.num_tasks) * -np.Inf, np.ones(self.num_states + self.num_tasks) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = task.device

        print("RL device: ", task.device)

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    def uniform_random_strategy(self, num_tasks, _):
        """A function for sampling tasks uniformly at random.

        Args:
            num_tasks (int): Total number of tasks.
            _ (object): Ignored by this sampling strategy.

        Returns:
            int: task id.

        """
        return random.randint(0, num_tasks - 1)

    def round_robin_strategy(self, num_tasks, last_task=None):
        """A function for sampling tasks in round robin fashion.

        Args:
            num_tasks (int): Total number of tasks.
            last_task (int): Previously sampled task.

        Returns:
            int: task id.

        """
        if last_task is None:
            return 0

        return (last_task + 1) % num_tasks

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

# Python CPU/GPU Class
class MetaVecTaskPython(VecTask):
    def set_task(self, task):
        self.task.this_task = task
        self.task_id = self.task_envs.index(task)
        self.task_info_index_lower = self.task_id * int(self.num_environments / self.num_tasks)
        self.task_info_index_upper = self.task_id * int(self.num_environments / self.num_tasks) + int(self.num_environments / self.num_tasks)

    def get_state(self):
        if self.mode == 'vanilla':
            state = torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        elif self.mode == 'add-onehot':
            state = torch.cat((torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), self._active_task_one_hot()), dim=1)

        return state

    def step(self, actions):

        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)
        if self.mode == 'vanilla':
            obs = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        elif self.mode == 'add-onehot':
            obs = torch.cat((torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), self._active_task_one_hot()), dim=1)

        return obs, self.task.rew_buf.to(self.rl_device), self.task.reset_buf.to(self.rl_device), self.task.extras

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)
        if self.mode == 'vanilla':
            obs = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        elif self.mode == 'add-onehot':
            obs = torch.cat((torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), self._active_task_one_hot()), dim=1)

        return obs

    def _active_task_one_hot(self):
        one_hot = torch.zeros((self.num_envs, self.num_tasks), device=self.rl_device)
        one_hot[:, self.task_id] = 1.0
        return one_hot
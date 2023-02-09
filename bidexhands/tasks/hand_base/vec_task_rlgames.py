# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from gym import spaces

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import torch
import numpy as np


# VecEnv Wrapper for RL training
class VecTask():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = task.device

        print("RL device: ", task.device)

        self.info = {}
        self.info['action_space'] = self.act_space
        self.info['observation_space'] = self.obs_space
        self.info['state_space'] = self.state_space
        self.info['agents'] = 1

    def has_action_masks(self):
        return False

    def get_number_of_agents(self):
        return self.num_environments

    def get_env_info(self):
        pass

    def seed(self, seed):
        pass

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        pass

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def get_number_of_agents(self):
        return self.num_agents

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

    def get_env_info(self):
        return self.info

# C++ CPU Class
class VecTaskCPU(VecTask):
    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations=clip_observations, clip_actions=clip_actions)
        self.sync_frame_time = sync_frame_time

    def step(self, actions):
        actions = actions.cpu().numpy()
        self.task.render(self.sync_frame_time)

        obs, rewards, resets, extras = self.task.step(np.clip(actions, -self.clip_actions, self.clip_actions))

        return (to_torch(np.clip(obs, -self.clip_obs, self.clip_obs), dtype=torch.float, device=self.rl_device),
                to_torch(rewards, dtype=torch.float, device=self.rl_device),
                to_torch(resets, dtype=torch.uint8, device=self.rl_device), [])

    def reset(self):
        actions = 0.01 * (1 - 2 * np.random.rand(self.num_envs, self.num_actions)).astype('f')

        # step the simulator
        obs, rewards, resets, extras = self.task.step(actions)

        return to_torch(np.clip(obs, -self.clip_obs, self.clip_obs), dtype=torch.float, device=self.rl_device)


# C++ GPU Class
class VecTaskGPU(VecTask):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations=clip_observations, clip_actions=clip_actions)

        self.obs_tensor = gymtorch.wrap_tensor(self.task.obs_tensor, counts=(self.task.num_envs, self.task.num_obs))
        self.rewards_tensor = gymtorch.wrap_tensor(self.task.rewards_tensor, counts=(self.task.num_envs,))
        self.resets_tensor = gymtorch.wrap_tensor(self.task.resets_tensor, counts=(self.task.num_envs,))

    def step(self, actions):
        self.task.render(False)
        actions_clipped = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_tensor = gymtorch.unwrap_tensor(actions_clipped)

        self.task.step(actions_tensor)

        return torch.clamp(self.obs_tensor, -self.clip_obs, self.clip_obs), self.rewards_tensor, self.resets_tensor, []

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))
        actions_tensor = gymtorch.unwrap_tensor(actions)

        # step the simulator
        self.task.step(actions_tensor)

        return torch.clamp(self.obs_tensor, -self.clip_obs, self.clip_obs)


# Python CPU/GPU Class
class RLgamesVecTaskPython(VecTask):

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        # Get obs dict mapped to correct device
        obs_dict = self._to_device({})

        # Clamp main obs buf and add it to obs dict
        obs_dict["obs"] = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        obs_dict["states"] = torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return obs_dict, self.task.rew_buf.to(self.rl_device), self.task.reset_buf.to(self.rl_device), self.task.extras

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)

        # Get obs dict mapped to correct device
        obs_dict = self._to_device({})

        # Clamp main obs buf and add it to obs dict
        obs_dict["obs"] = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        obs_dict["states"] = torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return obs_dict
    
    def _to_device(self, inp):
        """
        Maps all tensors in @inp to this object's device.

        Args:
            inp (tensor, iterable, dict): Any primitive data type that includes tensor(s)

        Returns:
            (tensor, iterable, dict): Same type as @inp, with all tensors mapped to self.rl_device
        """
        # Check all cases
        if isinstance(inp, torch.Tensor):
            inp = inp.to(self.rl_device)
        elif isinstance(inp, dict):
            for k, v in inp.items():
                inp[k] = self._to_device(v)
        else:
            # We assume that this is an iterable, so we loop over all entries
            for i, entry in enumerate(inp):
                inp[i] = self._to_device(entry)
        return inp
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
import random

class ReplayBuffer:

    def __init__(self, num_envs, replay_size, batch_size, num_transitions_per_env, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(replay_size, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(replay_size, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(replay_size, num_envs, 1, device=self.device)
        self.next_observations = torch.zeros(replay_size, num_envs, *obs_shape, device=self.device)
        self.actions = torch.zeros(replay_size, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(replay_size, num_envs, 1, device=self.device).byte()

        self.num_transitions_per_env = num_transitions_per_env
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.fullfill = False

        self.step = 0

    def add_transitions(self, observations, states, actions, rewards, next_obs ,dones):
        if self.step >= self.replay_size:
            #TODO: 有点bug 清不掉0 后续改下
            self.step = (self.step + 1) % self.replay_size
            # raise AssertionError("Rollout buffer overflow")
            self.fullfill = True

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.next_observations[self.step].copy_(next_obs)
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1


    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards[:self.step].mean()

    def mini_batch_generator(self, num_mini_batches):
        #TODO: 可以随机选择batch_size
        batch_size = self.batch_size
        mini_batch_size = batch_size // num_mini_batches
        batch = []
        # if self.sampler == "sequential":
        #     # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
        #     # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
        #     subset = SequentialSampler(range(batch_size))
        # elif self.sampler == "random":
        #     subset = SubsetRandomSampler(range(batch_size))
        for _ in range(num_mini_batches):
            if self.fullfill == True:
                subset = random.sample(range(self.replay_size), mini_batch_size)
            else:
                subset = random.sample(range(self.step), mini_batch_size)
        # batch = BatchSampler(subset, mini_batch_size, drop_last=True)
            batch.append(subset)
        return batch
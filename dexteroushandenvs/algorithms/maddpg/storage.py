import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
import random

class ReplayBuffer:

    def __init__(self, config, obs_shape, share_obs_shape, actions_shape, joint_actions_shape, device='cpu'):


        num_envs = config["n_rollout_threads"]
        num_transitions_per_env = config["replay_size"]
        self.batch_size = config["batch_size"]
        self.device = device
        self.sampler = config["sampler"] if config["sampler"] is not None else "random"


        # get dims of all agents' acitons
        joint_act_dim = 0
        for id in range(len(joint_actions_shape)):
            joint_act_dim += joint_actions_shape[id].shape[0]

        # Core
        self.obs = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        # share observation = share_obs
        self.share_obs = torch.zeros(num_transitions_per_env, num_envs, *share_obs_shape, device=self.device)

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        # next share observation = next_share_obs
        self.next_share_obs = torch.zeros(num_transitions_per_env, num_envs, *share_obs_shape,
                                              device=self.device)

        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        #share actions
        self.joint_actions = torch.zeros(num_transitions_per_env, num_envs, joint_act_dim, device=self.device)

        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.fullfill = False

        self.step = 0

    def add_transitions(self, observations, share_obs ,actions, joint_actions, rewards, next_obs ,next_state ,dones):
        if self.step >= self.num_transitions_per_env:
            #TODO: 有点bug 清不掉0 后续改下
            self.step = (self.step + 1) % self.num_transitions_per_env
            # raise AssertionError("Rollout buffer overflow")
            self.fullfill = True

        self.obs[self.step].copy_(observations)
        self.share_obs[self.step].copy_(share_obs)
        self.actions[self.step].copy_(actions)
        self.joint_actions[self.step].copy_(joint_actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.next_observations[self.step].copy_(next_obs)
        self.next_share_obs[self.step].copy_(next_state)
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1


    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

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
            if self.fullfill:
                subset = random.sample(range(self.num_transitions_per_env), mini_batch_size)
            else:
                subset = random.sample(range(self.step), mini_batch_size)
        # batch = BatchSampler(subset, mini_batch_size, drop_last=True)
            batch.append(subset)
        return batch


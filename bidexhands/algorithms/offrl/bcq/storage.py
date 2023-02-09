import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert(self, data_dir):
		self.state = np.load(data_dir+'states.npy')
		self.action = np.load(data_dir+'actions.npy')
		self.next_state = np.load(data_dir+'next_states.npy')
		self.reward = np.load(data_dir+'rewards.npy')
		self.not_done = 1. - np.load(data_dir+'dones.npy')
		self.size = self.state.shape[0]


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.normal as normal
import numpy as np

from copy import deepcopy

class TDNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, net_type):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.type = net_type
        self.nonlinearity_class = nn.ReLU
        self.layer_sizes = [256, 256]
        self.type = net_type

        if self.type == "Q":
            input_size = self.state_dim + self.action_dim
        else:
            input_size = self.state_dim

        layer_list = [nn.Linear(input_size, self.layer_sizes[0]), self.nonlinearity_class()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.nonlinearity_class())
        layer_list.append(nn.Linear(self.layer_sizes[-1], 1))

        for layer in layer_list:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim,max_action):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.nonlinearity_class = nn.ReLU
        self.layer_sizes = [256, 256]
        self.state_based_var = False

        layer_list = [nn.Linear(self.state_dim, self.layer_sizes[0]), self.nonlinearity_class()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.nonlinearity_class())

        for layer in layer_list:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

        self.features = nn.Sequential(*layer_list)
        self.means = nn.Linear(self.layer_sizes[-1], self.action_dim)
        nn.init.orthogonal_(self.means.weight, gain=np.sqrt(2))
        if self.state_based_var:
            self.log_vars = nn.Linear(self.layer_sizes[-1], self.action_dim)
            nn.init.orthogonal_(self.log_vars.weight, gain=np.sqrt(2))
        else:
            self.log_var = nn.Parameter(torch.zeros(self.action_dim))

    def forward_dist(self, states):
        features = self.features(states)
        means = self.max_action*torch.tanh(self.means(features))
        if self.state_based_var:
            stds = torch.exp(0.5 * self.log_vars(features))
        else:
            stds = torch.exp(0.5 * self.log_var)
        normal_dist = normal.Normal(means, stds)
        return torch.distributions.transformed_distribution.TransformedDistribution(normal_dist, [torch.distributions.transforms.TanhTransform(cache_size=1)])


class IQL_Model():
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        expectile=0.7,
        beta=3.0,
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.q_net_1 = TDNetwork(self.state_dim,self.action_dim,'Q').to(self.device)
        self.target_q_net_1 = deepcopy(self.q_net_1)
        self.q_net_2 = TDNetwork(self.state_dim,self.action_dim,'Q').to(self.device)
        self.target_q_net_2 = deepcopy(self.q_net_2)

        self.q_optimizer_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=3e-4)
        self.q_optimizer_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=3e-4)

        self.v_net = TDNetwork(self.state_dim,self.action_dim,'V').to(self.device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=3e-4)

        self.policy_net = Policy(self.state_dim,self.action_dim,self.max_action).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
            
        self.discount = discount
        self.expectile = expectile
        self.tau = tau
        self.beta = beta

    def expectile_loss(self, value, expectile_prediction):
        u = value - expectile_prediction
        return torch.mean(torch.abs(self.expectile * torch.square(u) + (nn.functional.relu(-1 * u) * u)))

    def square_loss(self, value, mean_prediction):
        return torch.mean(torch.square(value - mean_prediction))

    def L_V(self, states, actions):
        Q_input = torch.cat([states, actions], dim=-1)
        Q_1 = self.target_q_net_1.forward(Q_input)
        Q_2 = self.target_q_net_2.forward(Q_input)
        Q_values = torch.minimum(Q_1, Q_2)
        V_values = self.v_net.forward(states)
        return self.expectile_loss(Q_values, V_values)

    def L_Q(self, q_net, states, actions, rewards, next_states, terminals):
        Q_values = q_net.forward(torch.cat([states, actions], dim=-1)).squeeze(-1)
        V_values = self.v_net.forward(next_states).squeeze(-1) * (terminals)
        return self.square_loss(rewards + self.discount * V_values, Q_values)

    def target_update(self):
        for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
        for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)

    def TD_networks_update(self, states, actions, rewards, next_states, terminals):
        
        L_V = self.L_V(states, actions)
        self.v_optimizer.zero_grad()
        L_V.backward()
        self.v_optimizer.step()

        L_Q_1 = self.L_Q(self.q_net_1, states, actions, rewards, next_states, terminals)
        self.q_optimizer_1.zero_grad()
        L_Q_1.backward()
        self.q_optimizer_1.step()

        L_Q_2 = self.L_Q(self.q_net_2, states, actions, rewards, next_states, terminals)
        self.q_optimizer_2.zero_grad()
        L_Q_2.backward()
        self.q_optimizer_2.step()

        self.target_update()

    def L_pi(self, states, actions):

        Q_input = torch.cat([states, actions], dim=-1)
        Q_1 = self.target_q_net_1.forward(Q_input)
        Q_2 = self.target_q_net_2.forward(Q_input)
        Q_values = torch.minimum(Q_1, Q_2)
        V_values = self.v_net.forward(states)
        exp_advantages = torch.clip(torch.exp(self.beta * (Q_values - V_values)), max=100)
        action_log_probs = self.policy_net.forward_dist(states).log_prob(torch.clamp(actions, min=-0.99, max=0.99))
        return -1 * torch.mean(exp_advantages * action_log_probs)

    def policy_update(self, states, actions):

        L_pi = self.L_pi(states, actions)
        self.policy_optimizer.zero_grad()
        L_pi.backward()
        self.policy_optimizer.step()

    def select_action(self, state):     
        with torch.no_grad():
            action = self.max_action*torch.tanh(self.policy_net.means(self.policy_net.features(state)))
        return action

    def train(self,replay_buffer, interaction, batch_size=256):

        for update in range(interaction):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            self.TD_networks_update(state, action, reward, next_state, not_done)
            self.policy_update(state, action)


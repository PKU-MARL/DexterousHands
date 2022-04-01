import torch
import torch.nn as nn
from copy import deepcopy
from utils.util import get_gard_norm


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        a = nn.Linear(sizes[j], sizes[j + 1])
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class MLPActLayer(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh())
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return q  # Critical to ensure q has right shape.


class Actor(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU, device=torch.device("cuda:0")):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActLayer(obs_dim, act_dim, hidden_sizes, activation, act_limit)

        self.to(device)

    def act(self, obs):
        # with torch.no_grad():
        return self.pi(obs)


class Critic(nn.Module):

    def __init__(self, share_observation_space, share_action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU, device=torch.device("cuda:0")):
        super().__init__()

        # shared observation and acitons
        share_obs_dim = share_observation_space.shape[0]

        # multi agents
        share_act_dim = 0
        for id in range(len(share_action_space)):
            share_act_dim += share_action_space[id].shape[0]

        # build policy and value functions
        self.q = MLPQFunction(share_obs_dim, share_act_dim, hidden_sizes, activation)

        self.to(device)

    def get_value(self, share_obs, share_acts):
        """
        get the Qvalue
        :param share_obs: shared observation
        :param share_acts: a list of actions taken by all agents
        :return: q_vlaue:
        """

        # if len(share_acts) > 1:
        #     # squeeze the actions (sq_share_acts)
        #     sq_share_acts = share_acts[0]
        #     for i in range(1, len(share_acts)):
        #         sq_share_acts = torch.hstack((sq_share_acts, share_acts[i]))
        # else:
        #     sq_share_acts = share_acts

        q_value = self.q(share_obs, share_acts)

        return q_value


class MADDPG_policy:

    def __init__(self, config, obs_space, cent_obs_space, act_space, cent_act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = config["learning_rate"]
        self.hidden_size = config["hidden_size"]
        act_name = config["activation"]
        self.activation = get_activation(act_name)

        self.act_noise = config["act_noise"]

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.share_act_space = cent_act_space
        self.act_limit = act_space.high[0]

        self.actor = Actor(self.obs_space, self.act_space, self.hidden_size, self.activation, self.device)
        self.critic = Critic(self.share_obs_space, self.share_act_space, self.hidden_size, self.activation, self.device)

        self.actor_targ = deepcopy(self.actor)
        self.critic_targ = deepcopy(self.critic)
        # print(self.actor)
        # print(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.pi.parameters(),
                                                lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.q.parameters(),
                                                 lr=self.lr)

    def get_actions(self, obs, deterministic=False):
        """

        """
        actions = self.actor.act(obs)

        return actions.detach()

    def get_values(self, cent_obs, cent_acts):
        """
        Get value function predictions.

        """
        values, = self.critic(cent_obs, cent_acts)
        return values

    def act(self, obs, deterministic=False):

        if deterministic == True:

            actions = self.actor.act(obs)
        else:
            actions = self.actor.act(obs)
            actions = torch.clamp(actions + self.act_noise * torch.randn(actions.shape).to(self.device),
                                  -self.act_limit, self.act_limit)

        return actions.detach()


class MADDPG():

    def __init__(self,
                 config,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.num_agents = num_agents

        self.policy = policy
        self.num_learning_epochs = config["num_learning_epochs"]
        self.num_mini_batches = config["num_mini_batch"]
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        self.polyak = config["polyak"]
        self.max_grad_norm = config["max_grad_norm"]

    def cal_value_loss(self, data, nid):

        sobs = data[nid]['sobs']
        r = data[nid]['r']
        jact = data[nid]['jact']
        sobs2 = data[nid]['sobs2']
        d = data[nid]['done']

        q = self.policy[nid].critic.q(sobs, jact)

        # Bellman backup for Q functions
        with torch.no_grad():
            jact2 = []
            for vid in range(self.num_agents):
                pi_targ = self.policy[vid].actor_targ.pi(data[vid]['obs2'])

                jact2.append(pi_targ)

            jact2 = torch.cat(jact2, dim=-1)

            # Target Q-values
            q_pi_targ = self.policy[nid].critic.q(sobs2, jact2)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        value_loss = ((q - backup) ** 2).mean()

        return value_loss

    def cal_pi_loss(self, data, id):

        sobs = data[id]['sobs']

        jact = []
        # 得到其他智能体输出
        for pid in range(self.num_agents):
            action = self.policy[pid].actor.pi(data[id]['obs'])
            jact.append(action)

        jact = torch.cat(jact, dim=-1)

        q_pi = self.policy[id].critic.q(sobs, jact)
        pi_loss = -q_pi.mean()

        return pi_loss

    def ddpg_update(self, samples):
        value_loss = []
        policy_loss = []

        for id in range(self.num_agents):

            self.policy[id].critic_optimizer.zero_grad()

            loss_q = self.cal_value_loss(samples, id)

            loss_q.backward()
            nn.utils.clip_grad_norm_(self.policy[id].critic.parameters(), self.max_grad_norm)
            self.policy[id].critic_optimizer.step()

            # Record things
            value_loss.append(loss_q)

            # Next run one gradient descent step for pi.

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.policy[id].critic.q.parameters():
                p.requires_grad = False

            self.policy[id].actor_optimizer.zero_grad()
            loss_pi = self.cal_pi_loss(samples, id)
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.policy[id].actor.parameters(), self.max_grad_norm)
            self.policy[id].actor_optimizer.step()

            # Record things
            policy_loss.append(loss_pi)

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.policy[id].critic.q.parameters():
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.policy[id].critic.q.parameters(), self.policy[id].critic_targ.q.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

                for p, p_targ in zip(self.policy[id].actor.pi.parameters(), self.policy[id].actor_targ.pi.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        return value_loss, policy_loss

    def train(self, buffer):

        train_infos = []
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0

        batch = buffer[0].mini_batch_generator(self.num_mini_batches)

        learn_ep = 0
        for indices in batch:

            learn_ep += 1

            if learn_ep >= self.num_learning_epochs:
                break

            samples = []

            for id in range(self.num_agents):
                obs_batch = buffer[id].obs[indices]
                nextobs_batch = buffer[id].next_observations[indices]
                states_batch = buffer[id].share_obs[indices]
                next_states_batch = buffer[id].next_share_obs[indices]
                actions_batch = buffer[id].actions[indices]
                joint_actions_batch = buffer[id].joint_actions[indices]
                rewards_batch = buffer[id].rewards[indices]
                dones_batch = buffer[id].dones[indices]

                # TODO:  这里有点占空间 need 修改
                sample = {'obs': obs_batch,
                          'sobs': states_batch,
                          'act': actions_batch,
                          'jact': joint_actions_batch,
                          'r': rewards_batch,
                          'obs2': nextobs_batch,
                          'sobs2': next_states_batch,
                          'done': dones_batch}
                samples.append(sample)

            value_loss, policy_loss = self.ddpg_update(samples)

            for id in range(self.num_agents):
                train_info['value_loss'] += value_loss[id].item()
                train_info['policy_loss'] += policy_loss[id].item()
                train_infos.append(train_info)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for id in range(self.num_agents):
            for k in train_infos[id].keys():
                train_infos[id][k] /= num_updates

        return train_infos

    def prep_training(self):
        for id in range(self.num_agents):
            self.policy[id].actor.train()
            self.policy[id].critic.train()

    def prep_rollout(self):
        for id in range(self.num_agents):
            self.policy[id].actor.eval()
            self.policy[id].critic.eval()
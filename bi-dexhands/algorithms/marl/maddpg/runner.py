import os
import time
from gym.spaces import Space
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.maddpg import MADDPG as TrainAlgo
from algorithms.maddpg import MADDPG_policy as Policy
from algorithms.maddpg import ReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner:

    def __init__(self,
                 vec_env,
                 config,
                 model_dir=""
                 ):
        self.envs = vec_env
        self.eval_envs = vec_env
        # parameters
        self.env_name = vec_env.task.cfg["env"]["env_name"]
        self.algorithm_name = config["algorithm_name"]
        self.experiment_name = config["experiment_name"]
        # self.use_centralized_V = config["use_centralized_V"]
        # self.use_obs_instead_of_state = config["use_obs_instead_of_state"]
        self.num_env_steps = config["num_env_steps"]
        self.episode_length = config["episode_length"]
        self.n_rollout_threads = config["n_rollout_threads"]

        self.n_eval_rollout_threads = config["n_eval_rollout_threads"]
        # self.use_linear_lr_decay = config["use_linear_lr_decay"]
        self.hidden_size = config["hidden_size"]
        self.use_render = config["use_render"]
        # self.recurrent_N = config["recurrent_N"]
        # self.use_single_network = config["use_single_network"]
        # interval
        self.save_interval = config["save_interval"]
        self.use_eval = config["use_eval"]
        self.eval_interval = config["eval_interval"]
        self.eval_episodes = config["eval_episodes"]
        self.log_interval = config["log_interval"]

        self.seed = self.envs.task.cfg["seed"]
        self.model_dir = model_dir
        self.batch_size = config["batch_size"]
        self.warm_up = True

        self.num_agents = self.envs.num_agents
        self.device = self.envs.rl_device
        print(self.device)
        torch.autograd.set_detect_anomaly(True)

        self.run_dir = config["run_dir"]
        self.log_dir = str(
            self.run_dir + '/' + self.env_name + '/' + self.algorithm_name + '/logs_seed{}'.format(self.seed))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(
            self.run_dir + '/' + self.env_name + '/' + self.algorithm_name + '/models_seed{}'.format(self.seed))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id]
            # policy network
            po = Policy(config,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        self.envs.action_space,
                        device=self.device)
            self.policy.append(po)

        if self.model_dir != "":
            self.restore()

        self.trainer = []
        self.buffer = []


        self.trainer =  TrainAlgo(config, self.policy, self.num_agents,device=self.device)
        for agent_id in range(self.num_agents):
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id]

            bu = ReplayBuffer(config,
                               self.envs.observation_space[agent_id].shape,
                               share_observation_space.shape,
                               self.envs.action_space[agent_id].shape,
                               self.envs.action_space,
                               device = self.device,
                              )
            self.buffer.append(bu)

    def run(self):

        # warmup
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].states[0].copy_(share_obs[:, agent_id])
            self.buffer[agent_id].observations[0].copy_(obs[:, agent_id])


        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        train_episode_rewards = torch.tensor(train_episode_rewards, dtype=torch.float, device=self.device)

        for episode in range(episodes):

            done_episodes_rewards = []

            # train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]

            for step in range(self.episode_length):
                # Sample actions
                actions, joint_actions = self.collect(step)

                # Obser reward and next obs
                next_obs, next_share_obs, rewards, dones, infos, _ = self.envs.step(actions)

                dones_env = torch.all(dones, dim=1)

                reward_env = torch.mean(rewards, dim=1).flatten()

                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                data = obs, share_obs, rewards, next_obs, next_share_obs, actions, joint_actions, dones, infos

                # insert data into buffer
                self.insert(data)

                # reset observation
                obs = next_obs
                share_obs = next_share_obs

                #TODO: start training after a enough exploation
                # compute return and update network
                if self.buffer[0].step > self.batch_size:
                    self.warm_up = False

                if self.warm_up == False:
                    train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\nAlgo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                
                if self.warm_up == False:
                    self.log_train(train_infos, total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.mean(torch.stack(done_episodes_rewards))
                print("some episodes done, average rewards: ", aver_episode_rewards)
                self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                                         total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)


    def collect(self, step):
        action_collector = []

        for agent_id in range(self.num_agents):
            self.trainer.prep_rollout()
            action = self.trainer.policy[agent_id].act(self.buffer[agent_id].obs[step],deterministic = False)

            action_collector.append(action)

        # Todo:
        # [self.envs, agents, dim]
        # actions = torch.transpose(torch.stack(action_collector), 1, 0)

        # [self.envs, dim]  joint actions
        joint_acitons = torch.cat(action_collector,dim = -1)

        return action_collector,joint_acitons

    def insert(self, data):
        obs, share_obs, rewards, next_obs, next_share_obs, actions, joint_actions, dones, infos = data


        for agent_id in range(self.num_agents):
            self.buffer[agent_id].add_transitions(obs[:, agent_id], share_obs[:, agent_id], actions[agent_id],
                                                    joint_actions[:], rewards[:,agent_id],
                                                    next_obs[:,agent_id],
                                                    next_share_obs[:, agent_id], dones[:, agent_id])

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def train(self):

        # train the networks
        train_infos = self.trainer.train(self.buffer)

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer.policy[agent_id].actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer.policy[agent_id].critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):

            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            self.writter.add_scalars(k, {k: torch.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        while True:
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer.prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer.policy[agent_id].act(eval_obs[:, agent_id],deterministic=True)
                eval_actions_collector.append(eval_actions)

            eval_actions = eval_actions_collector

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = torch.all(eval_dones, dim=1)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(torch.sum(torch.cat(one_episode_rewards[eval_i]), dim=0))
                    one_episode_rewards[eval_i] = []

            if eval_episode >= self.eval_episodes:
                eval_episode_rewards = torch.cat(eval_episode_rewards,dim=-1)
                eval_env_infos = {'eval_average_episode_rewards': torch.mean(eval_episode_rewards),
                                  'eval_max_episode_rewards': torch.max(eval_episode_rewards)}
                print(eval_env_infos)
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(torch.mean(eval_episode_rewards)))
                break


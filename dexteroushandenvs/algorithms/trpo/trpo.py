from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algorithms.trpo import RolloutStorage


class TRPO:

    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 damping = 0.1,
                 cg_nsteps = 10,
                 max_kl = 1e-2,
                 max_num_backtrack= 10,
                 accept_ratio = 0.1,
                 step_fraction = 1,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
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

        self.schedule = schedule
        self.step_size = learning_rate

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric)
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # TRPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.damping = damping
        self.cg_nsteps = cg_nsteps
        self.max_kl= max_kl
        self.max_num_backtrack =max_num_backtrack
        self.accept_ratio = accept_ratio
        self.step_fraction = step_fraction

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

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            while True:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
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
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
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
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

                # Optimize policy
                # 1. find search direction for network parameter optimization, use conjugate gradient (CG)
                a_loss = -torch.squeeze(advantages_batch) * torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
                a_loss = a_loss.mean()
                g = torch.autograd.grad(a_loss, self.actor_critic.actor.parameters())
                flat_g = torch.cat([grad.view(-1) for grad in g]).data

                #KL
                kl = torch.sum(
                    sigma_batch - old_sigma_batch + (
                                torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                kl = torch.mean(kl)

                Av = lambda v: self.kl_hessian_times_vector(v,kl)
                step_dir = self.conjugate_gradient(Av, - flat_g, nsteps=self.cg_nsteps)

                # 2. find maximum stepsize along the search direction
                sAs = 0.5 * (step_dir * Av(step_dir)).sum(0)
                beta = torch.sqrt(2 * self.max_kl / sAs)
                full_step = (beta * step_dir).data.numpy()

                # 3. do line search along the found direction, with maximum change = full_step
                evaluate_policy = lambda x: self.get_aloss_logp(obs_batch,states_batch,actions_batch,advantages_batch,old_actions_log_prob_batch)
                a_old_loss, old_logp = evaluate_policy(None)
                success, new_params = self.line_search(evaluate_policy, full_step, flat_g,max_num_backtrack = self.max_num_backtrack,
                                                       accept_ratio = self.accept_ratio,step_fraction = self.step_fraction)
                self.set_pi_flat_params(new_params)
                a_new_loss, new_logp = evaluate_policy(None)

                surrogate_loss = a_new_loss

                # freeze the actor network
                for p in self.actor_critic.actor.parameters():
                    p.requires_grad = False

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Gradient step
                self.optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # recover the grad of actor network
                for p in self.actor_critic.actor.parameters():
                    p.requires_grad = True

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def conjugate_gradient(self,Av, b, nsteps, residual_tol=1e-10):
        """
        do conjugate gradient to find an approximated v such that A v = b

        ref: https://en.wikipedia.org/wiki/Conjugate_gradient_method
            The resulting algorithm

        :param Av: an oracle returns Av given v
        :param b: b, a vector
        :param nsteps: iterations
        :return: found v
        """
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for k in range(nsteps):
            av = Av(p)
            alpha = rdotr / torch.dot(p, av)
            x += alpha * p
            r -= alpha * av
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def line_search(self, evaluate_policy, full_step, grad, max_num_backtrack=10, accept_ratio=0.1,step_fraction =1.0):
        """
        do backtracking line search
        ref: https://en.wikipedia.org/wiki/Backtracking_line_search

        :param policy_net: policy net used to get initial params and set params before get_loss
        :param get_loss: get loss evaluation
        :param full_step: maximum stepsize, numpy.ndarray
        :param grad: initial gradient i.e. nabla f(x) in wiki
        :param max_num_backtrack: maximum iterations of backtracking
        :param accept_ratio: i.e. param c in wiki
        :return: a tuple (whether accepted at last, found optimal x)
        """
        # initial point
        x0 = self.get_pi_flat_params()
        # initial loss
        f0,olp = evaluate_policy(None)
        # step fraction
        alpha = step_fraction
        # expected maximum improvement, i.e. cm in wiki
        expected_improve = accept_ratio * (- full_step * grad).sum(0, keepdim=True)

        for count in range(max_num_backtrack):
            xnew = x0 + alpha * full_step
            self.set_pi_flat_params(xnew)
            fnew,olp = evaluate_policy(old_actions_log_prob_batch=olp)
            actual_improve = f0 - fnew
            if actual_improve > 0 and actual_improve > alpha * expected_improve:
                return True, xnew
            alpha *= 0.5
        return False, x0

    def kl_hessian_times_vector(self,v, kl):
        """
        return the product of KL's hessian and an arbitrary vector in O(n) time
        ref: https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/

        :param states: torch.Tensor(#samples, #d_state) used to calculate KL divergence on samples
        :param v: the arbitrary vector, torch.Tensor
        :return: (H + damping * I) dot v, where H = nabla nabla KL
        """
        # here, set create_graph=True to enable second derivative on function of this derivative
        grad_kl = torch.autograd.grad(kl, self.actor_critic.actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_kl_v = (flat_grad_kl * v).sum()
        grad_grad_kl_v = torch.autograd.grad(grad_kl_v, self.actor_critic.actor.parameters())
        flat_grad_grad_kl_v = torch.cat([grad.contiguous().view(-1) for grad in grad_grad_kl_v])

        return flat_grad_grad_kl_v + self.damping * v


    def set_pi_flat_params(self, flat_params):
        """
        set flat_params

        : param flat_params: Tensor
        """
        # flat_params = torch.Tensor(flat_params)
        prev_ind = 0
        for param in self.actor_critic.actor.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        self.old_log_prob = None

    def get_pi_flat_params(self):
        """
        get flat parameters
        returns numpy array
        """
        params = []
        for param in self.actor_critic.actor.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)

        # return flat_params.double().numpy()
        return flat_params

    def get_aloss_logp(self,obs_batch,states_batch,actions_batch,advantages_batch,old_actions_log_prob_batch):

        """

        :return:
        """
        actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(
            obs_batch,
            states_batch,
            actions_batch)

        a_loss = -torch.squeeze(advantages_batch) * torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
        a_loss = a_loss.mean()

        return a_loss, actions_log_prob_batch
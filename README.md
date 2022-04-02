# Bimanual Dexterous Manipulation via Reinforcement Learning
<img src="assets/image_folder/cover.jpg" width="1000" border="1"/>

**Bi-DexHands** provides a collection of bimanual dexterous manipulations tasks and reinforcement learning algorithms for solving them. 
Reaching human-level sophistication of hand dexterity and bimanual coordination remains an open challenge for modern robotics researchers. To better help the community study this problem, Bi-DexHands are developed with the following key features:
- **Isaac Efficiency**: Bi-DexHands is built within [Isaac Gym](https://developer.nvidia.com/isaac-gym); it supports running thousands of environments simultaneously. For example, on one NVIDIA RTX 3090 GPU, Bi-DexHands can reach **40,000+ mean FPS** by running  2,048  environments in parallel. 
- **RL/MARL Benchmark**: we provide the first bimanual manipulation task environment for RL and Multi-Agent RL practitioners, along with a comprehensive benchmark for SOTA continuous control model-free RL/MARL methods. See [example](#Demos)
- **Heterogeneous-agents Cooperation**: Agents (i.e., joitns, fingers, hands,...) in Bi-DexHands are genuinely heterogeneous; this is very different from common multi-agent environment such as [SMAC](https://github.com/oxwhirl/smac)  where agents can simply share parameters to solve the task. 
- **Task Generalization**: we introduce a variety of dexterous manipulation tasks (e.g., handover, lift up, throw, place, put...) as well as enormous target objects from the [YCB](https://rse-lab.cs.washington.edu/projects/posecnn/) and [SAPIEN](https://sapien.ucsd.edu/) dataset (>2,000 objects); this allows meta-RL and multi-task RL algorithms to be tested on the task generalization front. 

Bi-DexHands is becoming an important tool to evaluate the peroformance of RL-based solutions for robotics research. 

- [Installation](#Installation)
  - [Pre-requisites](#Installation)
- [Introduction to Bi-DexHands](#Introduction-to-Bi-DexHands)
  - [Demos](#Demos)
- [File Structure](#File-Structure)
- [Overview of Environments](./docs/environments.md)
- [Overview of Algorithms](./docs/algorithms.md)
- [Getting Started](#Getting-Started)
  - [Tasks]
  - [Training](#Training)
  - [Testing](#Testing)
  - [Plotting](#Plotting)
- [Enviroments Performance](#Enviroments-Performance)
  - [Figures](#Figures)
- [Building the Documentation](#Building-the-Documentation)
- [The Team](#The-Team)
- [License](#License)
<br></br>
****
## Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). **We currently support the `Preview Release 3` version of IsaacGym.**

### Pre-requisites

The code has been tested on Ubuntu 18.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `470.74` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments.
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 3 
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```

## Introduciton

This repository contains complex dexterous hand RL environments DexterousHandEnvs for the NVIDIA Isaac Gym high performance environments. DexterousHandEnvs is a very challenging dexterous hand manipulation environment for multi-agent reinforcement learning. We refer to some designs of existing multi-agent and dexterous hand environments, integrate their advantages, expand some new environments and unique features for multi-agent reinforcement learning. Our environments focus on the application of multi-agent algorithms to dexterous hand control, which is very challenging in traditional control algorithms. 

### Demos
<center class="half">
    <img src="assets/image_folder/0.gif" width="300"/><img src="assets/image_folder/3.gif"width="300"/>
</center>

For more demos please refer to [here](./docs/environments.md)

## Getting Started

### <span id="task">Tasks</span>

Source code for tasks can be found in `envs/tasks`. 

Until now we only suppose the following environments:

| Environments | Description | Demo     |
|  :----:  | :----:  | :----:  |
|ShadowHandOver| These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | <img src="assets/image_folder/0.gif" align="middle" width="550" border="1"/>    |
|ShadowHandCatchUnderarm| These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | <img src="assets/image_folder/4.gif" align="middle" width="140" border="1"/>    |
|ShadowHandCatchOver2Underarm| This environment is is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand | <img src="assets/image_folder/2.gif" align="middle" width="130" border="1"/>    |
|ShadowHandCatchAbreast| This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | <img src="assets/image_folder/1.gif" align="middle" width="130" border="1"/>    |
|ShadowHandCatchTwoCatchUnderarm| These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | <img src="assets/image_folder/sendpix0.gif" align="middle" width="130" border="1"/>    |


For more details about these environments please refer to [here](./docs/environments.md)

### Training

#### Gym-Like API

We provide a Gym-Like API that allows us to get information from the isaac-gym environment. Our single-agent Gym-Like wrapper is the code of the Isaacgym team used, and we have developed a multi-agent Gym-Like wrapper based on it:

```python
class MultiVecTaskPython(MultiVecTask):
    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        a_hand_actions = actions[0]
        for i in range(1, len(actions)):
            a_hand_actions = torch.hstack((a_hand_actions, actions[i]))
        actions = a_hand_actions

        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        hand_obs = []
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(torch.cat([obs_buf[:, :self.num_hand_obs], obs_buf[:, 2*self.num_hand_obs:]], dim=1))
        hand_obs.append(torch.cat([obs_buf[:, self.num_hand_obs:2*self.num_hand_obs], obs_buf[:, 2*self.num_hand_obs:]], dim=1))
        rewards = self.task.rew_buf.unsqueeze(-1).to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)

        sub_agent_obs = []
        agent_state = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(len(self.agent_index[0] + self.agent_index[1])):
            if i < len(self.agent_index[0]):
                sub_agent_obs.append(hand_obs[0])
            else:
                sub_agent_obs.append(hand_obs[1])

            agent_state.append(obs_buf)
            sub_agent_reward.append(rewards)
            sub_agent_done.append(dones)
            sub_agent_info.append(torch.Tensor(0))

        obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        reward_all = torch.transpose(torch.stack(sub_agent_reward), 1, 0)
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        info_all = torch.stack(sub_agent_info)

        return obs_all, state_all, reward_all, done_all, info_all, None

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions * 2], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)

        hand_obs = []
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs)
        hand_obs.append(torch.cat([obs_buf[:, :self.num_hand_obs], obs_buf[:, 2*self.num_hand_obs:]], dim=1))
        hand_obs.append(torch.cat([obs_buf[:, self.num_hand_obs:2*self.num_hand_obs], obs_buf[:, 2*self.num_hand_obs:]], dim=1))

        sub_agent_obs = []
        agent_state = []

        for i in range(len(self.agent_index[0] + self.agent_index[1])):
            if i < len(self.agent_index[0]):
                sub_agent_obs.append(hand_obs[0])
            else:
                sub_agent_obs.append(hand_obs[1])
            agent_state.append(obs_buf)

        obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs, state_all, None
```
#### RL/Multi-Agent RL API

Similar to the Gym-Like wrapper, we also provide single-agent and multi-agent RL algorithms respectively. In order to adapt to Isaacgym and speed up the running speed, all operations are done on the GPU using tensor, so there is no need to transfer data between the CPU and GPU, which greatly speeds up the operation.

We give an example to illustrate multi-agent RL APIs, which mainly refer to [https://github.com/cyanrain7/TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL):

```python
self.warmup()

start = time.time()
episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

train_episode_rewards = torch.zeros(1, self.n_rollout_threads, device=self.device)

for episode in range(episodes):
    if self.use_linear_lr_decay:
        self.trainer.policy.lr_decay(episode, episodes)

    done_episodes_rewards = []

    for step in range(self.episode_length):
        # Sample actions
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
        # Obser reward and next obs
        obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)
        dones_env = torch.all(dones, dim=1)
        reward_env = torch.mean(rewards, dim=1).flatten()
        train_episode_rewards += reward_env

        for t in range(self.n_rollout_threads):
            if dones_env[t]:
                done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                train_episode_rewards[:, t] = 0

        data = obs, share_obs, rewards, dones, infos, \
                values, actions, action_log_probs, \
                rnn_states, rnn_states_critic

        # insert data into buffer
        self.insert(data)

    # compute return and update network
    self.compute()
    train_infos = self.train()

    # post process
    total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
    # save model
    if (episode % self.save_interval == 0 or episode == episodes - 1):
        self.save()
```


#### Training Example

For example, if you want to train your first policy with ShadowHandOver task and PPO algorithm, run this line:

```bash
python train.py --task=ShadowHandOver --algo=ppo
```

To select an algorithm, pass `--algo=ppo/mappo/happo/hatrpo/...` 
as an argument. For example, if you want to use happo algorithm, run this line:

```bash
python train.py --task=ShadowHandOver --algo=hatrpo
``` 

Supported Single-Agent RL algorithms are listed below:

- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)

Supported Multi-Agent RL algorithms are listed below:

- [Heterogeneous-Agent Proximal Policy Optimization (HAPPO)](https://arxiv.org/pdf/2109.11251.pdf)
- [Heterogeneous-Agent Trust Region Policy Optimization (HATRPO)](https://arxiv.org/pdf/2109.11251.pdf)
- [Multi-Agent Proximal Policy Optimization (MAPPO)](https://arxiv.org/pdf/2103.01955.pdf)
- [Independent Proximal Policy Optimization (IPPO)](https://arxiv.org/pdf/2011.09533.pdf)
- [Multi-Agent Deep Deterministic Policy Gradient  (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)

For a brief introduction to these algorithms, please refer to [here](./docs/algorithms.md)

### Testing

The trained model will be saved to `logs/${Task Name}/${Algorithm Name}`folder.

To load a trained model and only perform inference (no training), pass `--test` 
as an argument, and pass `--model_dir` to specify the trained models which you want to load.
For single-agent reinforcement learning, you need to pass `--model_dir` to specify exactly what .pt model you want to load. An example is as follows:

```bash
python train.py --task=ShadowHandOver --model_dir=logs/shadow_hand_over/ppo/ppo_seed0/model_5000.pt --test
```

For multi-agent reinforcement learning, pass `--model_dir` to specify the path to the folder where all your agent model files are saved. An example is as follows:

```bash
python train.py --task=ShadowHandOver --model_dir=logs/shadow_hand_over/happo/models_seed0 --test
```

### Plotting

After training, you can convert all tfevent files into csv files and then try plotting the results.

```bash
# geenrate csv
$ python ./utils/logger/tools.py --root-dir ./logs/shadow_hand_over --refresh
# generate figures
$ python ./utils/logger/plotter.py --root-dir ./logs/shadow_hand_over --shaded-std --legend-pattern "\\w+"  --output-path=./logs/shadow_hand_over/figure.png
```

## Enviroment Performance

### Figures

For more figures please refer to [here](./docs/figures.md)

## Building the Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org) and the
readthedocs theme.

```bash
cd docs/
pip install -r requirements.txt
```
You can then build the documentation by running `make <format>` from the
`docs/` folder. Run `make` to get a list of all available output formats.

If you get a katex error run `npm install katex`.  If it persists, try
`npm install -g katex`

## The Team

DexterousHands is a PKU-MARL project under the leadership of Dr. [Yaodong Yang](https://www.yangyaodong.com/), it is currently maintained by [Yuanpei Chen](https://github.com/cypypccpy) and [Shengjie Wang](https://github.com/Shengjie-bob). 

It must be mentioned that in our development process, we mainly refer to the following two open source repositories: 

[https://github.com/NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) 

[https://github.com/cyanrain7/TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL) 


## License

DexterousHands has a Apache license, as found in the [LICENSE](LICENSE) file.

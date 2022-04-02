# Bimanual Dexterous Manipulation via Reinforcement Learning
<img src="assets/image_folder/cover.jpg" width="1000" border="1"/>

**Bi-DexMani** provides a collection of bimanual dexterous manipulations tasks and reinforcement learning algorithms for solving them. 
Reaching human-level sophistication of hand dexterity and bimanual coordination remains an open challenge for modern robotics researchers. To better help the community study this problem, Bi-DexMani are developed with the following key features:
- **Isaac Efficiency**: Bi-DexMani is built within [Isaac Gym](https://developer.nvidia.com/isaac-gym); it supports running thousands of environments simultaneously. For example, on one NVIDIA RTX 3090 GPU, Bi-DexMani can reach **40,000+ mean FPS** by running  2,048  environments in parallel. 
- **RL/MARL Benchmark**: we provide the first bimanual manipulation task environment for RL and Multi-Agent RL practitioners, along with a comprehensive benchmark for SOTA continuous control model-free RL/MARL methods. 
- **Heterogeneous-agents Cooperation**: Agents (i.e., joitns, fingers, hands,...) in Bi-DexMani are genuinely heterogeneous; this is very different from common multi-agent environment such as [SMAC](https://github.com/oxwhirl/smac)  where agents can simply share parameters to solve the task. 
- **Task Generalization**: we introduce a variety of dexterous manipulation tasks (e.g., handover, lift up, throw, place, put...) as well as enormous target objects from the [YCB](https://rse-lab.cs.washington.edu/projects/posecnn/) and [SAPIEN](https://sapien.ucsd.edu/) dataset (>2,000 objects); this allows meta-RL and multi-task RL algorithms to be tested on the task generalization front. 

Bi-DexMani is becoming an important tool to evaluate the peroformance of RL-based solutions for robotics research. 

- [Installation](#Installation)
  - [Pre-requisites](#Installation)
- [Introduction to Bi-DexMani](#Introduction-to-Bi-DexMani)
  - [What is Bi-DexMani](#What-is-Bi-DexMani)
  - [System design](#System-design)
  - [Summary](#Summary)
- [File Structure](#File-Structure)
- [Overview of Environments](./docs/environments.md)
- [Overview of Algorithms](./docs/algorithms.md)
- [Getting Started](#Getting-Started)
  - [Training](#Training)
  - [Testing](#Testing)
  - [Plotting](#Plotting)
- [Enviroments Performance](#Enviroments-Performance)
  - [Demos](#Demos)
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

## Introduction to Bi-DexMani

Bi-DexMani is a collection of environments and algorithms for learning bimanual dexterous manipulation. 

### System design

<img src="assets/image_folder/fig2-page-001.jpg" align="right" width="350" border="2"/>

The core of Bi-DexMani is to build up a learning framework for two shadow hands capable of diverse and general skills, such as **reach, throw, catch, pick and place**. To develop these skills, an agent must take cooperation between two hands into account. Bi-DexMani consists of three components, world, tasks and learning algorithms. Different worlds provide different  settings for robots, including the configuration of robotic hands and target objects. 

## Getting Started

### Training

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

## <span id="task">Tasks</span>

Source code for tasks can be found in `envs/tasks`. 

Until now we only suppose the following environments:

| Environments | ShadowHandOver | ShadowHandCatchUnderarm | ShadowHandTwoCatchUnderarm | ShadowHandCatchAbreast | ShadowHandOver2Underarm |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| Description | These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | This environment is is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand |
| Actions Type | Continuous | Continuous | Continuous | Continuous | Continuous |
| Total Action Num | 40    | 52    | 52    | 52    | 52    |
| Action Values     | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    |
| Observation Shape     | (num_envs, 2, 211)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    |
| Observation Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| State Shape     | (num_envs, 2, 398)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | 
| State Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| Rewards     | Rewards is the pose distance between object and goal. | Rewards is the pose distance between object and goal.    | Rewards is the pose distance between object and goal.    | Rewards is the pose distance between two object and  two goal, this means that both objects have to be thrown in order to be swapped over.    | Rewards is the pose distance between object and goal.   |
| Demo     | <img src="assets/image_folder/0.gif" align="middle" width="550" border="1"/>    | <img src="assets/image_folder/4.gif" align="middle" width="140" border="1"/>    | <img src="assets/image_folder/sendpix0.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/1.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/2.gif" align="middle" width="130" border="1"/>    |

For more details about these environments please refer to [here](./docs/environments.md)

## Enviroment Performance

### Demos

#### ShadowHandOver Environment
<img src="assets/image_folder/0.gif" align="center" width="700"/>

#### ShadowHandLiftUnderarm Environment
<img src="assets/image_folder/3.gif" align="center" width="700"/>

For more demos please refer to [here](./docs/environments.md)

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

It must be mentioned that in our development process, we mainly draw on the following two open source repositories: 

[https://github.com/NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) 

[https://github.com/cyanrain7/TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL) 


## License

DexterousHands has a Apache license, as found in the [LICENSE](LICENSE) file.

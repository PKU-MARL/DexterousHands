

<img src="assets/image_folder/cover.jpg" width="1000" border="1"/>

****
# Bimanual Dexterous Manipulation via Multi-Agent RL

### About this repository

Dexterous manipulaiton as a common but challenging task has attracted a great deal of interest in the field of robotics. Thanks to the intersection of reinforcement learning and robotics, previous study achieves a good performance on unimanual dexterous manipulaiton. However, how to balance between hand dexterity and bimanual coordination remains an open challenge. Therefore, we provided a novel benchmark for researchers to study machine intelligence. 

Bi-DexMani is a collection of environments and algorithms for learning bimanual dexterous manipulation. 

This repository contains complex dexterous hand RL environments DexterousHandEnvs for the NVIDIA Isaac Gym high performance environments. DexterousHandEnvs is a very challenging dexterous hand manipulation environment for multi-agent reinforcement learning. We refer to some designs of existing multi-agent and dexterous hand environments, integrate their advantages, expand some new environments and unique features for multi-agent reinforcement learning. Our environments focus on the application of multi-agent algorithms to dexterous hand control, which is very challenging in traditional control algorithms. 

The difficulty of our environment is not only reflected in the challenging task content but also reflected in the ultra-high-dimensional continuous space control. The state space dimension of each environment is up to 400 dimensions in total, and the action space dimension is up to 40 dimensions. A highlight of our environment is that we use five fingers and palms of each hand as a minimum unit, you can use each finger and palm as an agent, or combine any number of them as an agent by yourself.

:star2::star2:**Click [here](#task) to check the environment introduction!**:star2::star2:  

- [Installation](#Installation)
  - [Pre-requisites](#Installation)
- [Usage](#Usage)
  - [System design](#System-design)
  - [Running the benchmarks](#Running-the-benchmarks)
  - [Select an algorithm](#Select-an-algorithm)
  - [Loading trained models](#Loading-trained-models)
- [Tasks](#Tasks)
  - [HandCatchUnderarm Environments](#HandCatchUnderarm-Environments)
- [Building the Documentation](#Building-the-Documentation)
- [The Team](#The-Team)
- [License](#License)
<br></br>
****
## Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). **We currently support the `Preview Release 3` version of IsaacGym.**

### Pre-requisites

The code has been tested on Ubuntu 18.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `470` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments.
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 2 
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```

## Getting Started

### System design

<img src="assets/image_folder/fig2-page-001.jpg" align="right" width="350" border="2"/>

The core of Dual Dexterous Hands Manipulation is to build up a learning framework for two shadow hands capable of diverse and general skills, such as reach, throw, catch, pick and place. Apart from that, these skills should take the cooperation between two hands into account. To be specific, Bi-DetMani consists of three components, worlds, tasks and learning algorithms. Varying worlds provide a large number of basic settings for robots, including the configuration of robotic hands and objects. Meanwhile, a variety of tasks between two robotic hands make a single agent or multiple agents handle how to perform cooperative tasks.

### Running the benchmarks

To train your first policy with ShadowHandOver task and PPO algorithm, run this line:

```bash
python train.py --task=ShadowHandOver --algo=happo
```

### Select an algorithm

To select an algorithm, pass `--algo=ppo/mappo/happo/hatrpo/...` 
as an argument:

```bash
python train.py --task=ShadowHandOver --algo=hatrpo
``` 

Currently, we support the following algorithms: 

Single-Agent RL: **PPO, TRPO, SAC, TD3, DDPG** 

Multi-Agent RL: **IPPO, MAPPO, MADDPG, HATRPO, HAPPO**

### Loading trained models

<!-- Checkpoints are saved in the folder `models/`  -->

To load a trained checkpoint and only perform inference (no training), pass `--test` 
as an argument, and pass `--model_dir` to specify the trained models which you want to load.
For single-agent reinforcement learning, you need to pass `--model_dir` to specify exactly what .pt model you want to load. An example is as follows:

```bash
python train.py --task=ShadowHandOver --model_dir=logs/shadow_hand_over/ppo/ppo_seed0/model_5000.pt --test
```

For multi-agent reinforcement learning, pass `--model_dir` to specify the path to the folder where all your agent model files are saved. An example is as follows:

```bash
python train.py --task=ShadowHandOver --model_dir=logs/shadow_hand_over/happo/models_seed0 --test
```

## <span id="task">Tasks</span>

Source code for tasks can be found in `dexteroushandenvs/tasks`. 

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


### HandCatchUnderarm Environments
<img src="assets/image_folder/4.gif" align="middle" width="450" border="1"/>

Let's use ShadowHandCatchUnderarm environments as an example to show the detail. This environment have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchUnderarm`

#### <span id="obs2">Observation Space</span>


| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | shadow hand dof position |
| 24 - 47     | shadow hand dof velocity    |
| 48 - 71     | shadow hand dof force    |
| 72 - 136    | shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166     | shadow hand fingertip force, torque (5 x 6)    |
| 167 - 192     | actions    |
| 193 - 195     | shadow hand transition    |
| 196 - 198     | shadow hand orientation    |
| 199 - 205     | object pose   |
| 206 - 208     | object linear velocity    |
| 209 - 211     | object angle velocity    |
| 212 - 218     | goal pose    |
| 219 - 222     | goal rot - object rot   |

#### Action Space<span id="action2">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r2">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```


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

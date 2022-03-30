# DexterousHands

### About this repository

This repository contains complex dexterous hand RL environments DexterousHandEnvs for the NVIDIA Isaac Gym high performance environments. DexterousHandEnvs is a very challenging dexterous hand manipulation environment for multi-agent reinforcement learning. We refer to some designs of existing multi-agent and dexterous hand environments, integrate their advantages, expand some new environments and unique features for multi-agent reinforcement learning. Our environments focus on the application of multi-agent algorithms to dexterous hand control, which is very challenging in traditional control algorithms. 

The difficulty of our environment is not only reflected in the challenging task content but also reflected in the ultra-high-dimensional continuous space control. The state space dimension of each environment is up to 400 dimensions in total, and the action space dimension is up to 40 dimensions. A highlight of our environment is that we use five fingers and palms of each hand as a minimum unit, you can use each finger and palm as an agent, or combine any number of them as an agent by yourself.

:star2::star2:**Click [here](#task) to check the environment introduction!**:star2::star2:  

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

### Running the benchmarks

To train your first policy, run this line:

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

Single-Agent RL: PPO, TRPO, SAC, TD3, DDPG 

Multi-Agent RL: IPPO, MAPPO, MADDPG, HATRPO, HAPPO

<!-- ### Loading trained models // Checkpoints

Checkpoints are saved in the folder `models/` 

To load a trained checkpoint and only perform inference (no training), pass `--test` 
as an argument:

```bash
python train.py --task=ShadowHandOver --checkpoint=models/shadow_hand_over/ShadowHandOver.pth --test
``` -->

## <span id="task">Tasks</span>

Source code for tasks can be found in `dexteroushandenvs/tasks`. 

Until now we only suppose the following environments:

| Environments | ShadowHandOver | ShadowHandCatchUnderarm | ShadowHandCatchOverarm | ShadowHandTwoCatchUnderarm | ShadowHandCatchAbreast | ShadowHandOver2Underarm | ShadowHandLiftPot | ShadowHandOrientation |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| Description | These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | Similar to the HandCatchUnderArm environments but now the two hands are upright, and so the throwing/catching technique that has to be employed is different. | These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | This environment is is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand | Unlike most of these environment, this environment involve two unfixed hands and a pot in the table. We need to grab the handle of the pot with two hands and lift it up | Two-handed version of the classic shadowhand orientation environment, it need to rotate the object in the hand to a random target orientation in hand.
| Actions Type | Continuous | Continuous | Continuous | Continuous | Continuous | Continuous | Continuous | Continuous |
| Total Action Num | 40    | 52    | 52    | 52    | 52    | 52    | 52    | 52    |
| Action Values     | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    |
| Action Index and Description     | [detail](#action1)    | [detail](#action2)   | [detail](#action3)    | [detail](#action4)    | [detail](#action5)    | [detail](#action6)    | [detail](#action7)    | [detail](#action8)    |
| Observation Shape     | (num_envs, 2, 211)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    |
| Observation Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| Observation Index and Description     | [detail](#obs1)    | [detail](#obs2)   | [detail](#obs3)    | [detail](#obs4)    | [detail](#obs4)    | [detail](#obs4)    | [detail](#obs4)    | [detail](#obs4)    |
| State Shape     | (num_envs, 2, 398)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    |
| State Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| Rewards     | Rewards is the pose distance between object and goal. You can check out the details [here](#r1)| Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r3)    | Rewards is the pose distance between two object and  two goal, this means that both objects have to be thrown in order to be swapped over. You can check out the details [here](#r4)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    |
| Demo     | <img src="assets/image_folder/0.gif" align="middle" width="550" border="1"/>    | <img src="assets/image_folder/4.gif" align="middle" width="140" border="1"/>    | <img src="assets/image_folder/sendpix0.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/absxx-diyx0.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/1.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/1.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/1.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/1.gif" align="middle" width="130" border="1"/>    |

### HandOver Environments
<img src="assets/image_folder/0.gif" align="middle" width="450" border="1"/>


These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. To use the HandOver environment, pass `--task=ShadowHandOver`

#### <span id="obs1">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | shadow hand dof position |
| 24 - 47     | shadow hand dof velocity    |
| 48 - 71     | shadow hand dof force    |
| 72 - 136    | shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166     | shadow hand fingertip force, torque (5 x 6)    |
| 167 - 186     | actions    |
| 187 - 193     | object pose   |
| 194 - 196     | object linear velocity    |
| 197 - 199     | object angle velocity    |
| 200 - 206     | goal pose    |
| 207 - 210     | goal rot - object rot   |

#### Action Space<span id="action1">Action Space</span>
The shadow hand has 24 joints, 20 actual drive joints and 4 underdrive joints. So our Action is the joint Angle value of the 20 dimensional actuated joint.
| Index | Description |
|  ----  | ----  |
| 0 - 19 | shadow hand actuated joint |

#### <span id="r1">Rewards</span>
Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```
Object receives a large (250) bonus when it reaches goal. When the ball drops, it will reset the environment, but will not receive a penalty.


### HandCatchUnderarm Environments
<img src="assets/image_folder/4.gif" align="middle" width="450" border="1"/>

These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchUnderarm`

#### #### <span id="obs2">Observation Space</span>


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
Object receives a large (250) bonus when it reaches goal. When the ball drops, it will reset the environment, but will not receive a penalty.

### HandCatchOverarm Environments
<img src="assets/image_folder/sendpix0.jpg" align="middle" width="450" border="1"/>

Similar to the HandCatchUnderArm environments but now the two hands are upright, and so the throwing/catching technique that has to be employed is different. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchOverarm`

#### #### <span id="obs3">Observation Space</span>


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

#### Action Space<span id="action3">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r3">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
# Orientation alignment for the cube in hand and goal cube
quat_diff = quat_mul(object_rot, quat_conjugate(target_rot)
reward = (0.3 - goal_dist - quat_diff)
```
Object receives a large (250) bonus when it reaches goal. When the ball drops, it will reset the environment, but will not receive a penalty.

### TwoObjectCatch Environments
<img src="assets/image_folder/sendpix2.jpg" align="middle" width="450" border="1"/>

These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). This is necessary since each object's goal can only be reached by the other hand. To use the HandCatchUnderarm environment, pass `--task=ShadowHandTwoCatchUnderarm`

#### #### <span id="obs4">Observation Space</span>


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
| 199 - 205     | object1 pose   |
| 206 - 208     | object1 linear velocity    |
| 210 - 212     | object1 angle velocity    |
| 213 - 219     | goal1 pose    |
| 220 - 223     | goal1 rot - object1 rot   |
| 224 - 230     | object2 pose   |
| 231 - 233     | object2 linear velocity    |
| 234 - 236     | object2 angle velocity    |
| 237 - 243     | goal2 pose    |
| 244 - 247     | goal2 rot - object2 rot   |

#### Action Space<span id="action4">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r4">Rewards</span>

Rewards is the pose distance between two object and  two goal, this means that both objects have to be thrown in order to be swapped over. The specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
goal_another_dist = torch.norm(target_another_pos - object_another_pos, p=2, dim=-1)

# Orientation alignment for the cube in hand and goal cube
quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

quat_another_diff = quat_mul(object_another_rot, quat_conjugate(target_another_rot))
rot_another_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_another_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist)) + torch.exp(-0.2*(goal_another_dist * dist_reward_scale + rot_another_dist))
```
Object receives a large (250) bonus when it reaches goal. When the ball drops, it will reset the environment, but will not receive a penalty.

### HandCatchAbreast Environments
<img src="assets/image_folder/1.gif" align="middle" width="450" border="1"/>

These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchAbreast`

#### #### <span id="obs5">Observation Space</span>


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

#### Action Space<span id="action5">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r5">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```
Object receives a large (250) bonus when it reaches goal. When the ball drops, it will reset the environment, but will not receive a penalty.

## Citing

Please cite this work as:

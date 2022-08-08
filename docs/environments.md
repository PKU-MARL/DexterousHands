## Environments

This repository contains complex dexterous hand RL environments bi-dexhands for the NVIDIA Isaac Gym high performance environments. bi-dexhands is a very challenging dexterous hand manipulation environment for multi-agent reinforcement learning. We refer to some designs of existing multi-agent and dexterous hand environments, integrate their advantages, expand some new environments and unique features for multi-agent reinforcement learning. Our environments focus on the application of multi-agent algorithms to dexterous hand control, which is very challenging in traditional control algorithms. 

We provide a detailed description of the environment here. For single-agent reinforcement learning, all states and actions are used. For multi-agent reinforcement learning, we use the most common one: each hand as an agent, and a total of two agents as an example to illustrate.

The observation of all tasks is composed of three parts: the state values of the left and right hands, and the information of objects and target. The state values of the left and right hands were the same for each task, including hand joint and finger positions, velocity, and force information. The state values of the object and goal are different for each task, which we will describe in the following. [Here](#obs_normal) gives the specific information of the left-hand and right-hand state values. Note that the observation is slightly different in the HandOver task due to the fixed base.

#### <span id="obs_normal">Observation space of dual shadow hands</span>
| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | right shadow hand dof position |
| 24 - 47 |	right shadow hand dof velocity |
| 48 - 71 | right shadow hand dof force |
| 72 - 136 |	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166 |	right shadow hand fingertip force, torque (5 x 6) |
| 167 - 169 |	right shadow hand base position |
| 170 - 172 |	right shadow hand base rotation |
| 173 - 198 |	right shadow hand actions |
| 199 - 222 |	left shadow hand dof position |
| 223 - 246 |	left shadow hand dof velocity |
| 247 - 270 |   left shadow hand dof force |
| 271 - 335 |	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 336 - 365 |	left shadow hand fingertip force, torque (5 x 6) |
| 366 - 368 |	left shadow hand base position |
| 369 - 371 |	left shadow hand base rotation |
| 372 - 397 |	left shadow hand actions |

### HandOver Environments
<img src="../assets/image_folder/0.gif" align="middle" width="450" border="1"/> 

This environment consists of two shadow hands with palms facing up, opposite each other, and an object that needs to be passed. In the beginning, the object will fall randomly in the area of the shadow hand on the right side. Then the hand holds the object and passes the object to the other hand. Note that the base of the hand is fixed. More importantly, the hand which holds the object initially can not directly touch the target, nor can it directly roll the object to the other hand, so the object must be thrown up and stays in the air in the process. There are 398-dimensional observations and 40-dimensional actions in the task. Additionally, the reward function is related to the pose error between the object and the target. When the pose error gets smaller, the reward increases dramatically. To use the HandOver environment, pass `--task=ShadowHandOver`

#### <span id="obs1">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 373 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 374 - 380 |	object pose |
| 381 - 383 |	object linear velocity |
| 384 - 386 |	object angle velocity |
| 387 - 393 |	goal pose |
| 394 - 397 |	goal rot - object rot |

#### <span id="action1">Action Space</span>

| Index | Description |
|  ----  | ----  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 39 |	left shadow hand actuated joint |

#### <span id="r1">Rewards</span>
Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```

### HandCatchUnderarm Environments
<img src="../assets/image_folder/hand_catch_underarm.gif" align="middle" width="450" border="1"/>

In this problem, two shadow hands with palms facing upwards are controlled to pass an object from one palm to the other. What makes it more difficult than the Handover problem is that the hands' translation and rotation degrees of freedom are no longer frozen but are added into the action space. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchUnderarm`

#### <span id="obs2">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |

#### <span id="action2">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r2">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```

### HandCatchOver2Underarm Environments
<img src="../assets/image_folder/2.gif" align="middle" width="450" border="1"/>

This environment is like made up of half Hand Over and Catch Underarm, the object needs to be thrown from the vertical hand to the palm-up hand. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchOver2Underarm`

#### <span id="obs3">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |

#### <span id="action3">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r3">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
# Orientation alignment for the cube in hand and goal cube
quat_diff = quat_mul(object_rot, quat_conjugate(target_rot)

rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```

### TwoCatchUnderarm Environments
<img src="../assets/image_folder/two_catch.gif" align="middle" width="450" border="1"/>

This environment is similar to Catch Underarm, but with an object in each hand and the corresponding goal on the other hand. Therefore, the environment requires two objects to be thrown into the other hand at the same time, which requires a higher manipulation technique than the environment of a single object. To use the HandCatchUnderarm environment, pass `--task=ShadowHandTwoCatchUnderarm`

#### <span id="obs4">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object1 pose |
| 405 - 407 |	object1 linear velocity |
| 408 - 410 |	object1 angle velocity |
| 411 - 417 |	goal1 pose |
| 418 - 421 |	goal1 rot - object1 rot |
| 422 - 428 |	object2 pose |
| 429 - 431 |	object2 linear velocity |
| 432 - 434 |	object2 angle velocity |
| 435 - 441 |	goal2 pose |
| 442 - 445 |	goal2 rot - object2 rot |

#### <span id="action4">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

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

### HandCatchAbreast Environments
<img src="../assets/image_folder/1.gif" align="middle" width="450" border="1"/>

This environment consists of two shadow hands placed side by side in the same direction and an object that needs to be passed. Compared with the previous environment which is more like passing objects between the hands of two people, this environment is designed to simulate the two hands of the same person passing objects, so different catch techniques are also required and require more hand translation and rotation techniques. To use the HandCatchAbreast environment, pass `--task=ShadowHandCatchAbreast`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```

### Lift Underarm Environments
<img src="../assets/image_folder/3.gif" align="middle" width="450" border="1"/>

This environment requires grasping the pot handle with two hands and lifting the pot to the designated position. This environment is designed to simulate the scene of lift in daily life and is a  practical skill. To use the Lift Underarm environment, pass `--task=ShadowHandLiftUnderarm`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	object right handle position |
| 425 - 427 |	object left handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the left handle, the distance from the right hand to the right handle, and the distance from the object to the target point.

```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

right_hand_dist = torch.norm(pot_right_handle_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(pot_left_handle_pos - left_hand_pos, p=2, dim=-1)

right_hand_dist_rew = right_hand_dist
left_hand_dist_rew = left_hand_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = torch.where(right_hand_dist < 0.08,
                    torch.where(left_hand_dist < 0.08,
                                    3*(0.985 - goal_dist), up_rew), up_rew)

reward = 0.2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### Door Open Outward/Door Close Inward Environments
<img src="../assets/image_folder/open_outward.gif" align="middle" width="450" border="1"/>

These two environments require a closed/opened door to be opened/closed and the door can only be pushed outward or initially open inward. Both these two environments only need to do the push behavior, so it is relatively simple. To use the Door Open Outward/Door Close Inward environment, pass `--task=ShadowHandDoorOpenOutward/DoorCloseInward`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	object right handle position |
| 425 - 427 |	object left handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the left handle, the distance from the right hand to the right handle, and the distance from the object to the target point.

```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

right_hand_dist = torch.norm(door_right_handle_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(door_left_handle_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(door_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(door_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
# if door open outward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)
# if door close inward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                1 - torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### Door Open Inward/Door Close Outward Environments
<img src="../assets/image_folder/door_open_inward.gif" align="middle" width="450" border="1"/>

These two environments also require a closed/opened door to be opened/closed and the door can only be pushed inward or initially open outward, but because they can't complete the task by simply pushing, which need to catch the handle by hand and then open or close it, so it is relatively difficult. To use the Door Open Outward/Door Close Inward environment, pass `--task=ShadowHandDoorOpenInward/DoorCloseOutward`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	object right handle position |
| 425 - 427 |	object left handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the left handle, the distance from the right hand to the right handle, and the distance from the object to the target point.

```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

right_hand_dist = torch.norm(door_right_handle_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(door_left_handle_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(door_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(door_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
# if door close outward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                1 - torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

reward = 6 - right_hand_dist_rew - left_hand_dist_rew + up_rew

# if door open inward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### Bottle Cap Environments
<img src="../assets/image_folder/bottle_cap.gif" align="middle" width="450" border="1"/>

This environment involves two hands and a bottle, we need to hold the bottle with one hand and open the bottle cap with the other hand. This skill requires the cooperation of two hands to ensure that the cap does not fall. To use the Bottle Cap environment, pass `--task=ShadowHandBottleCap`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	bottle pose |
| 405 - 407 |	bottle linear velocity |
| 408 - 410 |	bottle angle velocity |
| 411 - 413 |	bottle cap position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward also consists of three parts: the distance from the left hand to the bottle cap, the distance from the right hand to the bottle, and the distance between the bottle and bottle cap.

```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

right_hand_dist = torch.norm(bottle_cap_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(bottle_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(bottle_cap_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(bottle_cap_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(bottle_cap_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(bottle_cap_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(bottle_cap_pos - right_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_dist

up_rew = torch.zeros_like(right_hand_dist_rew)

up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

reward = 2.0 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### Push Block Environments
<img src="../assets/image_folder/static_image/push_block.jpg" align="middle" width="450" border="1"/>

This environment involves two hands and two blocks, we need to use both hands to reach and push the block to the desired goal separately. This is a relatively simple task. To use the Push Block environment, pass `--task=ShadowHandPushBlock`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	block1 pose |
| 405 - 407 |	block1 linear velocity |
| 408 - 410 |	block1 angle velocity |
| 411 - 413 |	block1 position |
| 414 - 416 |	block2 position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to block1, the distance from the right hand to block2, and the distance between the block and desired goal.

```python
left_goal_dist = torch.norm(target_pos - block_left_handle_pos, p=2, dim=-1)
right_goal_dist = torch.norm(target_pos - block_right_handle_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(block_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(block_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(block_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(block_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(block_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(block_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = torch.exp(-10*right_hand_finger_dist)
left_hand_dist_rew = torch.exp(-10*left_hand_finger_dist)

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = (torch.exp(-10*left_goal_dist) + torch.exp(-10*right_goal_dist)) * 2

reward = right_hand_dist_rew + left_hand_dist_rew + up_rew
```

### Swing Cup Environments
<img src="../assets/image_folder/static_image/swing_cup.jpg" align="middle" width="450" border="1"/>

This environment involves two hands and a dual handle cup, we need to use two hands to hold and swing the cup together. To use the Swing Cup environment, pass `--task=ShadowHandSwingCup`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	cup pose |
| 405 - 407 |	cup linear velocity |
| 408 - 410 |	cup angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	cup right handle position |
| 425 - 427 |	cup left handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the cup's left handle, the distance from the right hand to the cup's right handle, and the rotating distance between the cup and desired goal.

```python
right_hand_finger_dist = (torch.norm(cup_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(cup_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(cup_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(cup_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(cup_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(cup_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(cup_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(cup_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(cup_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(cup_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale - 1

up_rew = torch.zeros_like(rot_rew)
up_rew = torch.where(right_hand_finger_dist < 0.4,
                    torch.where(left_hand_finger_dist < 0.4,
                                    rot_rew, up_rew), up_rew)
    
reward = - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### Open Scissors Environments
<img src="../assets/image_folder/static_image/scissors.jpg" align="middle" width="450" border="1"/>

This environment involves two hands and scissors, we need to use two hands to open the scissors. To use the Open Scissors environment, pass `--task=ShadowHandOpenScissors`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	scissors pose |
| 405 - 407 |	scissors linear velocity |
| 408 - 410 |	scissors angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	scissors right handle position |
| 425 - 427 |	scissors left handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the scissors' left handle, the distance from the right hand to the scissors' right handle, and the target angle at which the scissors need to be opened.

```python
right_hand_finger_dist = (torch.norm(scissors_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(scissors_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(scissors_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(scissors_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(scissors_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(scissors_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(scissors_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(scissors_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(scissors_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(scissors_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = torch.where(right_hand_finger_dist < 0.7,
                torch.where(left_hand_finger_dist < 0.7,
                    (0.59 + object_dof_pos[:, 0]) * 5, up_rew), up_rew)

reward = 2 + up_rew - right_hand_dist_rew - left_hand_dist_rew
```

### Re Orientation Environments
<img src="../assets/image_folder/static_image/re_orientation.jpg" align="middle" width="450" border="1"/>

This environment involves two hands and two objects. Each hand holds an object and we need to reorient the object to the target orientation. To use the Re Orientation environment, pass `--task=ShadowHandReOrientation`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object1 pose |
| 405 - 407 |	object1 linear velocity |
| 408 - 410 |	object1 angle velocity |
| 411 - 417 |	goal1 pose |
| 418 - 421 |	goal1 rot - object1 rot |
| 422 - 428 |	object2 pose |
| 429 - 431 |	object2 linear velocity |
| 432 - 434 |	object2 angle velocity |
| 435 - 441 |	goal2 pose |
| 442 - 445 |	goal2 rot - object2 rot |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left object to the left object goal, the distance from the right object to the right object goal, and the distance between the object and desired goal.

```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
goal_another_dist = torch.norm(target_another_pos - object_another_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

quat_another_diff = quat_mul(object_another_rot, quat_conjugate(target_another_rot))
rot_another_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_another_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist * dist_reward_scale + goal_another_dist * dist_reward_scale
rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale + 1.0/(torch.abs(rot_another_dist) + rot_eps) * rot_reward_scale

reward = dist_rew + rot_rew + action_penalty * action_penalty_scale
```

### Open Pen Cap Environments
<img src="../assets/image_folder/static_image/pen_cap.jpg" align="middle" width="450" border="1"/>

This environment involves two hands and a pen, we need to use two hand to open the pen cap. To use the Open Pen Cap environment, pass `--task=ShadowHandPen`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	pen pose |
| 405 - 407 |	pen linear velocity |
| 408 - 410 |	pen angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	pen right handle position |
| 425 - 427 |	pen left handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the pen body, the distance from the right hand to the pen cap, and the distance between the pen body and pen cap.

```python
right_hand_finger_dist = (torch.norm(pen_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(pen_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(pen_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(pen_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(pen_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(pen_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(pen_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(pen_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(pen_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(pen_left_handle_pos - left_hand_th_pos, p=2, dim=-1))


right_hand_dist_rew = torch.exp(-10 * right_hand_finger_dist)
left_hand_dist_rew = torch.exp(-10 * left_hand_finger_dist)

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = torch.where(right_hand_finger_dist < 0.75,
                torch.where(left_hand_finger_dist < 0.75,
                    torch.norm(pen_right_handle_pos - pen_left_handle_pos, p=2, dim=-1) * 5 - 0.8, up_rew), up_rew)

reward = up_rew + right_hand_dist_rew + left_hand_dist_rew
```

### Switch Environments
<img src="../assets/image_folder/static_image/switch.jpg" align="middle" width="450" border="1"/>

This environment involves dual hands and a bottle, we need to use dual hand fingers to press the desired button. To use the Switch environment, pass `--task=ShadowHandSwitch`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	switch1 pose |
| 405 - 407 |	switch1 linear velocity |
| 408 - 410 |	switch1 angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	switch1 position |
| 425 - 427 |   switch2 position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the left switch, the distance from the right hand to the right switch, and the distance between the button and button's desired goal.

```python
right_hand_finger_dist = (torch.norm(pen_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(pen_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(pen_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(pen_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(pen_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(pen_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(pen_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(pen_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(pen_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(pen_left_handle_pos - left_hand_th_pos, p=2, dim=-1))


right_hand_dist_rew = torch.exp(-10 * right_hand_finger_dist)
left_hand_dist_rew = torch.exp(-10 * left_hand_finger_dist)

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = (1.4-(switch_right_handle_pos[:, 2] + switch_left_handle_pos[:, 2])) * 50

reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### Stack Block Environments
<img src="../assets/image_folder/static_image/stack_block.jpg" align="middle" width="450" border="1"/>

This environment involves dual hands and two blocks, and we need to stack the block as a tower. To use the Stack Block environment, pass `--task=ShadowHandBlockStack`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	block1 pose |
| 405 - 407 |	block1 linear velocity |
| 408 - 410 |	block1 angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	block1 position |
| 425 - 427 |   block2 position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to block1, the distance from the right hand to block2, and the distance between the block and desired goal.

```python
stack_pos1 = target_pos.clone()
stack_pos2 = target_pos.clone()

stack_pos1[:, 1] -= 0.1
stack_pos2[:, 1] -= 0.1
stack_pos1[:, 2] += 0.05

goal_dist1 = torch.norm(stack_pos1 - block_left_handle_pos, p=2, dim=-1)
goal_dist2 = torch.norm(stack_pos2 - block_right_handle_pos, p=2, dim=-1)

right_hand_dist = torch.norm(block_right_handle_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(block_left_handle_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(block_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(block_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(block_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(block_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(block_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(block_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                    (0.24 - goal_dist1 - goal_dist2) * 2, up_rew), up_rew)

stack_rew = torch.zeros_like(right_hand_dist_rew)
stack_rew = torch.where(goal_dist2 < 0.07,
                torch.where(goal_dist1 < 0.07,
                    (0.05-torch.abs(stack_pos1[:, 2] - block_left_handle_pos[:, 2])) * 50 ,stack_rew),stack_rew)

reward = 1.5 - right_hand_dist_rew - left_hand_dist_rew + up_rew + stack_rew
```

### Pour Water Environments
<img src="../assets/image_folder/static_image/pour_water.jpg" align="middle" width="450" border="1"/>

This environment involves two hands and a bottle, we need to Hold the kettle with one hand and the bucket with the other hand, and pour the water from the kettle into the bucket. In the practice task in Isaac Gym, we use many small balls to simulate the water. To use the Pour Water environment, pass `--task=ShadowHandPourWater`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	kettle pose |
| 405 - 407 |	kettle linear velocity |
| 408 - 410 |	kettle angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |
| 422 - 424 |	kettle handle position |
| 425 - 427 |   bucket position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 22 |	right shadow hand base translation |
| 23 - 25 |	right shadow hand base rotation |
| 26 - 45 |	left shadow hand actuated joint |
| 46 - 48 |	left shadow hand base translation |
| 49 - 51 |	left shadow hand base rotation |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the bucket, the distance from the right hand to the kettle, and the distance between the kettle spout and desired goal.

```python
right_hand_finger_dist = (torch.norm(kettle_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(kettle_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(kettle_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(kettle_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(kettle_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(bucket_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(bucket_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(bucket_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(bucket_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                        + torch.norm(bucket_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = torch.where(right_hand_finger_dist < 0.7,
                torch.where(left_hand_finger_dist < 0.7,
                                0.5 - torch.norm(bucket_handle_pos - kettle_spout_pos, p=2, dim=-1) * 2, up_rew), up_rew)

reward = 1 + up_rew - right_hand_dist_rew - left_hand_dist_rew
```
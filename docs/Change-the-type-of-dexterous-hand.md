# How to change the type of dexterous hand
The default dexterous hand used by Bi-DexHands framework is the Shadow Hand. We know there is more type of dexterous hands than the shadow hand like allegro hand, trifinger, et al, and supporting other dexterous hands helps to advance research and community development. 

We provide an [example](../bi-dexhands/tasks/allegro_hand_over.py) for using allegro hand to complete Hand Over task. Pass `--task=AllegroHandOver --algo=ppo`to use this environment.

<div align=center>
<img src="../assets/image_folder/allegro_hand_over.gif" align="center" width="600"/>
</div> 

## A brief introduction to the main modification
Prepare the [urdf/mjcf](../assets/urdf/allegro_hand_model/urdf/allegro_hand_r.urdf) model file for the new dexterous hand and import it into the environment: 
```python
allegro_hand_asset_file = "urdf/allegro_hand_model/urdf/allegro_hand_r.urdf"
allegro_hand_another_asset_file = "urdf/allegro_hand_model/urdf/allegro_hand_r.urdf"
```
Modify the action dimension: 
```python
if self.is_multi_agent:
    self.num_agents = 2
    self.cfg["env"]["numActions"] = 16
    
else:
    self.num_agents = 1
    self.cfg["env"]["numActions"] = 32
```
DoF-related dimensions in observation: 
```python
self.obs_buf[:, action_obs_start:action_obs_start + 16] = self.actions[:, :16]

# another_hand
another_hand_start = action_obs_start + 16
...
self.obs_buf[:, action_another_obs_start:action_another_obs_start + 16] = self.actions[:, 16:]

obj_obs_start = action_another_obs_start + 16  # 144
```
The index of the target DoF position of the hands (Action): 
```python
self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :16],
                                                        self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                            self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])

self.cur_targets[:, self.actuated_dof_indices + 16] = scale(self.actions[:, 16:32],
                                                        self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
self.cur_targets[:, self.actuated_dof_indices + 16] = self.act_moving_average * self.cur_targets[:,
                                                                                            self.actuated_dof_indices + 16] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
self.cur_targets[:, self.actuated_dof_indices + 16] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 16],
                                                                self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
```
And always need to modify the initial position: 
```python
allegro_hand_start_pose = gymapi.Transform()
allegro_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.571, -1.571, 0)

allegro_another_hand_start_pose = gymapi.Transform()
allegro_another_hand_start_pose.p = gymapi.Vec3(0, -0.4, 0.5)
allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(-1.571, -1.571, 0)
```
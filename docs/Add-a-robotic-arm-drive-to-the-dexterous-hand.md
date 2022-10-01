# How to add a robotic arm drive to the dexterous hand 
Using a robotic arm drive at the base of the dexterous hand not only makes the environment more realistic, but also an important step for sim2real transfer. Because the dynamics of the flying hand is not easy to match reality -- and attaching the robotic arm can minimize the reality gap by adjusting the dynamics physics parameters of the arm. 

We provide an example of an allegro hand to add a X-Arm 6 to complete the Catch Underarm task. Pass `--task=AllegroHandCatchUnderarm --algo=ppo`to use this environment.

<div align=center>
<img src="../assets/image_folder/allegro_catch_underarm.gif" align="center" width="600"/>
</div> 

## A brief introduction to the main modification
First, combine the model file [(urdf/mjcf)](../assets/urdf/xarm_description/urdf/xarm6.urdf) of the robotic arm with the dexterous hand and load the model in the environment:
```python
allegro_hand_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"
allegro_hand_another_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"
```

We directly use DoF velocity as the action of the robotic arm, and other methods such as solve IK can also be used: 
```python
# x-arm control
targets = self.prev_targets[:, :6] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, :6]
self.cur_targets[:, :6] = tensor_clamp(targets,
                        self.allegro_hand_dof_lower_limits[:6], self.allegro_hand_dof_upper_limits[:6])

self.cur_targets[:, self.actuated_dof_indices + 6] = scale(self.actions[:, 6:22],
                                                        self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
self.cur_targets[:, self.actuated_dof_indices + 6] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 6],
                                                                self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])

targets = self.prev_targets[:, 22:28] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, 22:28]
self.cur_targets[:, 22:28] = tensor_clamp(targets,
                        self.allegro_hand_dof_lower_limits[:6], self.allegro_hand_dof_upper_limits[:6])

self.cur_targets[:, self.actuated_dof_indices + 28] = scale(self.actions[:, 28:44],
                                                        self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
self.cur_targets[:, self.actuated_dof_indices + 28] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 28],
                                                                self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
```

Other similar methods of dealing with different DoFs can refer to [this](../docs/Change-the-type-of-dexterous-hand.md).

The initial DoF of the robotic arm can be adjusted here:
```python
# create some wrapper tensors for different slices
self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
self.allegro_hand_default_dof_pos[:6] = torch.tensor([0, 0, -1, 3.14, 0.57, 3.14], dtype=torch.float, device=self.device)
```

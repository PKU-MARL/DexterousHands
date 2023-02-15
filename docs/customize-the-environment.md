# Customizing Environment

- [How to replace objects in existing tasks with items from YCB and Sapien datasets](#How_to_replace_objects_in_existing_tasks_with_items_from_YCB_and_Sapien_datasets)
  - [YCB dataset](#YCB_dataset)
  - [Sapien dataset](#Sapien_dataset)
- [How to create your own task](#How_to_create_your_own_task)

## How to replace objects in existing tasks with items from YCB and Sapien datasets 

We use objects from the YCB dataset for our object-catching environment, and objects from the Sapien dataset for other environments. According to our reward design, the reward required for object-catching tasks is only the distance from the object to the target point, so the objects in the YCB dataset can be directly replaced. Other tasks generally require a left-hand and right-hand gripping place as an auxiliary, so the Sapien dataset has additional position information that needs to be defined by ourselves. Below I will introduce these two methods separately: 

### <span id="ycb">YCB dataset</span>
**Applicable environment: ShadowHandOver, ShadowHandCatchUnderarm, ShadowHandCatchOver2Underarm, ShadowHandCatchAbreast, ShadowHandTwoCatchUndearm.**

To replace objects in the environment with objects in the YCB dataset, just put the model into the asset and modify the object path in the environment py file. Take **ShadowHandOver** as an example: if you want to replace the objects in **ShadowHandOver** with banana in the YCB dataset, you only need to put the model file of banana into [assets/urdf/ycb/011_banana/011_banana.urdf](.assets/urdf/ycb/011_banana/011_banana.urdf), add the model path in [bi-dexhands/tasks/shadow_hand_over.py](.bi-dexhands/tasks/shadow_hand_over.py):

```python
self.asset_files_dict = {
    "block": "urdf/objects/cube_multicolor.urdf",
    "egg": "mjcf/open_ai_assets/hand/egg.xml",
    "pen": "mjcf/open_ai_assets/hand/pen.xml",
    "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
    "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
    "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
    "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
}
```

and change the "egg" in [config](bi-dexhands/cfg/shadow_hand_over.yaml) to "ycb/banana". 

#### possible problems: 
If the object is too large for the hand to put down, consider reducing the size of the object: 
```python
self.gym.set_actor_scale(env_ptr, object_handle, 0.3)
```
Or adjust the initialization positions of objects and targets: 
```python
object_start_pose = gymapi.Transform()
object_start_pose.p = gymapi.Vec3()
object_start_pose.r = gymapi.Quat().from_euler_zyx(0, -0.7, 0)
pose_dx, pose_dy, pose_dz = -0, 0.45, -0.0

object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz
```
If the object has multiple joints, aggregate body errors and dof force acquisition problems may occur. 
Aggregate body errors simply add the correct number of bodies here: 
```python
# compute aggregate size
max_agg_bodies = self.num_shadow_hand_bodies * 2 + 2
max_agg_shapes = self.num_shadow_hand_shapes * 2 + 2
```
The dof force needs to be reduced here according to the specific amount of dof (Take [ShadowHandDoorCloseInward](bi-dexhands/tasks/shadow_hand_door_close_inward.py) as an example, it have 4 extra dof from objects): 
```python
dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs * 2 + 4)
self.dof_force_tensor = self.dof_force_tensor[:, :48]
```

### Sapien dataset:
**Applicable environment: Others**
To use the sapien dataset, compared to the YCB dataset, the most important thing is to deal with multiple dofs and the problem of auxiliary grasping points. 
Below I have divided it into three steps. The first step is to modify the model files in the Sapien dataset, and deal with the problems of aggregate and dof force, as in the [above YCB dataset](#ycb). 
The second step is to define the position of the auxiliary grasping point used in the reward function. This step needs to be used in conjunction with debug function. Take **ShadowHandDoorOpenOutward** as an example. First, we need to get the pose of the body of the model: 
```python
self.door_left_handle_pos = self.rigid_body_states[:, 26 * 2 + 3, 0:3]
self.door_left_handle_rot = self.rigid_body_states[:, 26 * 2 + 3, 3:7]

self.door_right_handle_pos = self.rigid_body_states[:, 26 * 2 + 2, 0:3]
self.door_right_handle_rot = self.rigid_body_states[:, 26 * 2 + 2, 3:7]

```
Then, we can use this pose as the origin to translate the three axes (x, y, and z) to obtain the pose of the auxiliary point: 
```python
self.door_left_handle_pos = self.rigid_body_states[:, 26 * 2 + 3, 0:3]
self.door_left_handle_rot = self.rigid_body_states[:, 26 * 2 + 3, 3:7]
self.door_left_handle_pos = self.door_left_handle_pos + quat_apply(self.door_left_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.5)
self.door_left_handle_pos = self.door_left_handle_pos + quat_apply(self.door_left_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.39)
self.door_left_handle_pos = self.door_left_handle_pos + quat_apply(self.door_left_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.04)

self.door_right_handle_pos = self.rigid_body_states[:, 26 * 2 + 2, 0:3]
self.door_right_handle_rot = self.rigid_body_states[:, 26 * 2 + 2, 3:7]
self.door_right_handle_pos = self.door_right_handle_pos + quat_apply(self.door_right_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.5)
self.door_right_handle_pos = self.door_right_handle_pos + quat_apply(self.door_right_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.39)
self.door_right_handle_pos = self.door_right_handle_pos + quat_apply(self.door_right_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.04)
```
The whole process can use the debug function to view the position in the render to determine the length of the translation, you need to set `True` in [config](bi-dexhands/cfg/shadow_hand_door_open_outward.yaml): 
```python
enableDebugVis: True
```
Then fill in the pose you want to visualize here: 
```python
enableDebugVis: True
```
The third step is to deal with the reset problem of the new object. When we reset the environment, because the dof of each object is different, the reset of the dof part needs to be redefined according to different objects. For example XX, the object's dof is 2: 
```python
self.object_dof_pos[env_ids, :] = to_torch([0, 0], device=self.device)
self.goal_object_dof_pos[env_ids, :] = to_torch([0, 0], device=self.device)
self.object_dof_vel[env_ids, :] = to_torch([0, 0], device=self.device)
self.goal_object_dof_vel[env_ids, :] = to_torch([0, 0], device=self.device)

...

self.prev_targets[env_ids, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + 2] = to_torch([0, 0], device=self.device)
self.cur_targets[env_ids, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + 2] = to_torch([0, 0], device=self.device)
self.prev_targets[env_ids, self.num_shadow_hand_dofs*2 + 2:self.num_shadow_hand_dofs*2 + 2*2] = to_torch([0, 0], device=self.device)
self.cur_targets[env_ids, self.num_shadow_hand_dofs*2 + 2:self.num_shadow_hand_dofs*2 + 2*2] = to_torch([0, 0], device=self.device)
```

## How to create your own task

Creating your own task is very simple and only takes three steps. Take **ShadowHandOver** as an example. First, create a `.py` file in the `bi-dexhands/tasks/` folder, which contains an environment class, you can refer to the existing task:
```
class ShadowHandOver(BaseTask):
...
```
Create a `.yaml` file in the `bi-dexhands/cfg/` folder:
```
bi-dexhands/cfg/ShadowHandOver.yaml
```
Next, retrieve the config file and add the class name of the new task in the format: 
```python
if args.task in ["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandTwoCatchUnderarm",      "ShadowHandCatchAbreast", "ShadowHandReOrientation",
                "ShadowHandLiftUnderarm", "ShadowHandCatchOver2Underarm", "ShadowHandBottleCap", "ShadowHandDoorCloseInward", "ShadowHandDoorOpenInward",
                "ShadowHandDoorOpenInward", "ShadowHandDoorOpenOutward", "ShadowHandKettle", "ShadowHandPen", "ShadowHandBlockStack", "ShadowHandSwitch",
                "ShadowHandPushBlock", "ShadowHandSwingCup", "ShadowHandGraspAndPlace", "ShadowHandScissors"]:
        
```
Finally, retrieve the task class in `bi-dexhands/utils/parse_task.py`:
```python
from bidexhands.tasks.shadow_hand_over import ShadowHandOver
```
Regarding how to design the content of the task, you can mainly refer to the methods [mentioned above](#ycb). After learning about isaacgym and RL, I believe that you can easily design your own tasks. If there is still something you don't understand, welcome to submit an issue to discuss with us :-).

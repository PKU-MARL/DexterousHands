# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

from .shadow_hand_meta_mt4_task_info import obtain_task_info, compute_hand_reward

class ShadowHandMetaMT4(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.task_envs = ["hand_over", "catch_underarm", "two_catch_underarm", "catch_abreast"]
        self.this_task = "catch_underarm"

        self.num_tasks = len(self.task_envs)
        self.num_each_envs = self.cfg["env"]["numEnvs"]

        self.asset_files_dict = {
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pot": "mjcf/pot/mobility.urdf",
            "door": "mjcf/door/mobility.urdf",
        }

        self.num_asset = len(self.asset_files_dict)
        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 424
        }
        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = 'z'

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.a_fingertips = ["robot1:ffdistal", "robot1:mfdistal", "robot1:rfdistal", "robot1:lfdistal", "robot1:thdistal"]

        self.hand_center = ["robot1:palm"]

        self.num_fingertips = len(self.fingertips) * 2

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 26
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 52

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg, is_meta=True, task_num=self.num_tasks)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            tem_dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor)
            self.dof_force_tensor = torch.zeros((self.num_envs, 48), dtype=torch.float, device=self.device)
            pointer = 0
            for i in range(self.num_envs):
                self.dof_force_tensor[i, :] = tem_dof_force_tensor[pointer:pointer+48]
                pointer += 48 + self.object_dof[i] * 2

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.num_total_dof = self.dof_state.shape[0]

        tem_shadow_hand_dof = torch.zeros((self.num_envs, 48, 2), dtype=torch.float, device=self.device)
        pointer = 0
        for i in range(self.num_envs):
            tem_shadow_hand_dof[i, :, :] = self.dof_state[pointer:pointer+48, :]
            pointer += 48 + self.object_dof[i] * 2
        self.saved_dof_state = self.dof_state.clone()

        self.shadow_hand_dof_state = tem_shadow_hand_dof[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.shadow_hand_another_dof_state = tem_shadow_hand_dof[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2]
        self.shadow_hand_another_dof_pos = self.shadow_hand_another_dof_state[..., 0]
        self.shadow_hand_another_dof_vel = self.shadow_hand_another_dof_state[..., 1]

        self.total_rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor)
        self.num_total_rigid_body = self.total_rigid_body_tensor.shape[0]
        self.rigid_body_states = torch.zeros((self.num_envs, 52, 13), dtype=torch.float, device=self.device)
        pointer = 0
        for i in range(self.num_envs):
            self.rigid_body_states[i, :, :] = self.total_rigid_body_tensor[pointer:pointer+52, :]
            pointer += 52 + self.object_rigid_body[i] * 2 + 1

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 

        self.prev_targets = torch.zeros((self.num_envs, 48), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, 48), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, 52, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, 52, 3), device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        shadow_hand_another_asset_file = "mjcf/open_ai_assets/hand/shadow_hand1.xml"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        shadow_hand_another_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_another_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        a_relevant_tendons = ["robot1:T_FFJ1c", "robot1:T_MFJ1c", "robot1:T_RFJ1c", "robot1:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        a_tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_another_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
            for rt in a_relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_another_asset, i) == rt:
                    a_tendon_props[i].limit_stiffness = limit_stiffness
                    a_tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        self.gym.set_asset_tendon_properties(shadow_hand_another_asset, a_tendon_props)
        
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        shadow_hand_another_dof_props = self.gym.get_asset_dof_properties(shadow_hand_another_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_assets = []
        goal_assets = []  
        another_object_assets = []
        another_goal_assets = []      
        table_assets = []
        shadow_hand_start_poses = []
        another_shadow_hand_start_poses = []
        object_start_poses = []
        goal_start_poses = []
        another_object_start_poses = []
        another_goal_start_poses = []
        table_start_poses = []

        for task_env in self.task_envs:
            hand_start_pose, another_hand_start_pose, object_pose, goal_pose, another_object_pose, another_goal_pose, table_pose_dim, object_asset_options, object_type = obtain_task_info(task_env)
                        
            goal_asset_options = gymapi.AssetOptions()
            goal_asset_options.disable_gravity = True
            another_object_asset_options = gymapi.AssetOptions()
            another_object_asset_options.density = 500
            if task_env != "two_catch_underarm":
                another_object_asset_options.disable_gravity = True

            shadow_hand_start_pose = gymapi.Transform()
            shadow_hand_start_pose.p = gymapi.Vec3(hand_start_pose[0], hand_start_pose[1], hand_start_pose[2])
            shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(hand_start_pose[3], hand_start_pose[4], hand_start_pose[5])

            shadow_another_hand_start_pose = gymapi.Transform()
            shadow_another_hand_start_pose.p = gymapi.Vec3(another_hand_start_pose[0], another_hand_start_pose[1], another_hand_start_pose[2])
            shadow_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(another_hand_start_pose[3], another_hand_start_pose[4], another_hand_start_pose[5])

            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3(object_pose[0], object_pose[1], object_pose[2])
            object_start_pose.r = gymapi.Quat().from_euler_zyx(object_pose[3], object_pose[4], object_pose[5])

            goal_start_pose = gymapi.Transform()
            goal_start_pose.p = gymapi.Vec3(goal_pose[0], goal_pose[1], goal_pose[2])
            goal_start_pose.r = gymapi.Quat().from_euler_zyx(goal_pose[3], goal_pose[4], goal_pose[5])

            another_object_start_pose = gymapi.Transform()
            another_object_start_pose.p = gymapi.Vec3(another_object_pose[0], another_object_pose[1], another_object_pose[2])
            another_object_start_pose.r = gymapi.Quat().from_euler_zyx(another_object_pose[3], another_object_pose[4], another_object_pose[5])

            another_goal_start_pose = gymapi.Transform()
            another_goal_start_pose.p = gymapi.Vec3(another_goal_pose[0], another_goal_pose[1], another_goal_pose[2])
            another_goal_start_pose.r = gymapi.Quat().from_euler_zyx(another_goal_pose[3], another_goal_pose[4], another_goal_pose[5])

            # create table asset
            table_dims = gymapi.Vec3(table_pose_dim[6], table_pose_dim[7], table_pose_dim[8])
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.flip_visual_attachments = False
            asset_options.collapse_fixed_joints = True
            asset_options.disable_gravity = True
            asset_options.thickness = 0.001

            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pose_dim[0], table_pose_dim[1], table_pose_dim[2])
            table_pose.r = gymapi.Quat().from_euler_zyx(table_pose_dim[3], table_pose_dim[4], table_pose_dim[5])

            object_assets.append(self.gym.load_asset(self.sim, asset_root, self.asset_files_dict[object_type], object_asset_options))
            goal_assets.append(self.gym.load_asset(self.sim, asset_root, self.asset_files_dict[object_type], goal_asset_options))
            another_object_assets.append(self.gym.load_asset(self.sim, asset_root, self.asset_files_dict[object_type], another_object_asset_options))
            another_goal_assets.append(self.gym.load_asset(self.sim, asset_root, self.asset_files_dict[object_type], goal_asset_options))
            table_assets.append(self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options))

            shadow_hand_start_poses.append(shadow_hand_start_pose)
            another_shadow_hand_start_poses.append(shadow_another_hand_start_pose)
            object_start_poses.append(object_start_pose)
            goal_start_poses.append(goal_start_pose)
            another_object_start_poses.append(another_object_start_pose)
            another_goal_start_poses.append(another_goal_start_pose)
            table_start_poses.append(table_pose)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 2 + 2
        max_agg_shapes = self.num_shadow_hand_shapes * 2 + 2

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.goal_init_state = []
        self.another_object_init_state = []
        self.another_goal_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.another_object_indices = []
        self.another_goal_object_indices = []
        self.table_indices = []
        self.object_dof = []
        self.object_rigid_body = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.fingertip_another_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_another_asset, name) for name in self.a_fingertips]

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
            for ft_a_handle in self.fingertip_another_handles:
                self.gym.create_asset_force_sensor(shadow_hand_another_asset, ft_a_handle, sensor_pose)
        
        for env_id in range(self.num_tasks):
            for i in range(self.num_each_envs):
                index = i + self.num_each_envs * env_id
                # create env instance
                env_ptr = self.gym.create_env(
                    self.sim, lower, upper, num_per_row
                )

                if self.aggregate_mode >= 1:
                    self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

                # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
                shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_poses[env_id], "hand", index, 0, 0)
                shadow_hand_another_actor = self.gym.create_actor(env_ptr, shadow_hand_another_asset, another_shadow_hand_start_poses[env_id], "another_hand", index, 0, 0)
                
                self.hand_start_states.append([shadow_hand_start_poses[env_id].p.x, shadow_hand_start_poses[env_id].p.y, shadow_hand_start_poses[env_id].p.z,
                                            shadow_hand_start_poses[env_id].r.x, shadow_hand_start_poses[env_id].r.y, shadow_hand_start_poses[env_id].r.z, shadow_hand_start_poses[env_id].r.w,
                                            0, 0, 0, 0, 0, 0])
                
                self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
                hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
                self.hand_indices.append(hand_idx)

                self.gym.set_actor_dof_properties(env_ptr, shadow_hand_another_actor, shadow_hand_another_dof_props)
                another_hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_another_actor, gymapi.DOMAIN_SIM)
                self.another_hand_indices.append(another_hand_idx)            

                # randomize colors and textures for rigid body
                num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
                hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
                
                for n in self.agent_index[0]:
                    colorx = random.uniform(0, 1)
                    colory = random.uniform(0, 1)
                    colorz = random.uniform(0, 1)
                    for m in n:
                        for o in hand_rigid_body_index[m]:
                            self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(colorx, colory, colorz))
                for n in self.agent_index[1]:                
                    colorx = random.uniform(0, 1)
                    colory = random.uniform(0, 1)
                    colorz = random.uniform(0, 1)
                    for m in n:
                        for o in hand_rigid_body_index[m]:
                            self.gym.set_rigid_body_color(env_ptr, shadow_hand_another_actor, o, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(colorx, colory, colorz))
                    # gym.set_rigid_body_texture(env, actor_handles[-1], n, gymapi.MESH_VISUAL,
                    #                            loaded_texture_handle_list[random.randint(0, len(loaded_texture_handle_list)-1)])

                # create fingertip force-torque sensors
                if self.obs_type == "full_state" or self.asymmetric_obs:
                    self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
                    self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_another_actor)
                
                # add object
                object_handle = self.gym.create_actor(env_ptr, object_assets[env_id], object_start_poses[env_id], "object", index, 0, 0)
                self.object_init_state.append([object_start_poses[env_id].p.x, object_start_poses[env_id].p.y, object_start_poses[env_id].p.z,
                                            object_start_poses[env_id].r.x, object_start_poses[env_id].r.y, object_start_poses[env_id].r.z, object_start_poses[env_id].r.w,
                                            0, 0, 0, 0, 0, 0])
                object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
                self.object_indices.append(object_idx)

                self.object_dof.append(self.gym.get_asset_dof_count(object_assets[env_id]) * 2)
                self.object_rigid_body.append(self.gym.get_asset_rigid_body_count(object_assets[env_id]) * 2)

                # add goal object
                goal_handle = self.gym.create_actor(env_ptr, goal_assets[env_id], goal_start_poses[env_id], "goal_object", index + self.num_envs, 0, 0)
                self.goal_init_state.append([goal_start_poses[env_id].p.x, goal_start_poses[env_id].p.y, goal_start_poses[env_id].p.z,
                                            goal_start_poses[env_id].r.x, goal_start_poses[env_id].r.y, goal_start_poses[env_id].r.z, goal_start_poses[env_id].r.w,
                                            0, 0, 0, 0, 0, 0])                
                goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
                self.goal_object_indices.append(goal_object_idx)

                # add another object
                object_another_handle = self.gym.create_actor(env_ptr, another_object_assets[env_id], another_object_start_poses[env_id], "another_object", index, 0, 0)
                self.another_object_init_state.append([another_object_start_poses[env_id].p.x, another_object_start_poses[env_id].p.y, another_object_start_poses[env_id].p.z,
                                            another_object_start_poses[env_id].r.x, another_object_start_poses[env_id].r.y, another_object_start_poses[env_id].r.z, another_object_start_poses[env_id].r.w,
                                            0, 0, 0, 0, 0, 0])

                object_another_idx = self.gym.get_actor_index(env_ptr, object_another_handle, gymapi.DOMAIN_SIM)
                self.another_object_indices.append(object_another_idx)

                # add another goal object
                goal_another_handle = self.gym.create_actor(env_ptr, another_goal_assets[env_id], another_goal_start_poses[env_id], "another_goal_object", index + self.num_envs * 2, 0, 0)
                self.another_goal_init_state.append([another_goal_start_poses[env_id].p.x, another_goal_start_poses[env_id].p.y, another_goal_start_poses[env_id].p.z,
                                            another_goal_start_poses[env_id].r.x, another_goal_start_poses[env_id].r.y, another_goal_start_poses[env_id].r.z, another_goal_start_poses[env_id].r.w,
                                            0, 0, 0, 0, 0, 0])      
                another_goal_object_idx = self.gym.get_actor_index(env_ptr, goal_another_handle, gymapi.DOMAIN_SIM)
                self.another_goal_object_indices.append(another_goal_object_idx)

                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

                # add table
                table_handle = self.gym.create_actor(env_ptr, table_assets[env_id], table_start_poses[env_id], "table", index, 0, 0)
                self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
                table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
                self.table_indices.append(table_idx)

                if self.aggregate_mode > 0:
                    self.gym.end_aggregate(env_ptr)

                self.envs.append(env_ptr)
                self.shadow_hands.append(shadow_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.another_object_init_state = to_torch(self.another_object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.another_goal_init_state = to_torch(self.another_goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.goal_states = self.goal_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.another_goal_states = self.another_goal_init_state.clone()
        self.another_goal_pose = self.another_goal_states[:, 0:7]
        self.another_goal_pos = self.another_goal_states[:, 0:3]
        self.another_goal_rot = self.another_goal_states[:, 3:7]
        self.another_goal_init_state = self.another_goal_states.clone()

        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.fingertip_another_handles = to_torch(self.fingertip_another_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.another_object_indices = to_torch(self.another_object_indices, dtype=torch.long, device=self.device)
        self.another_goal_object_indices = to_torch(self.another_goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.another_object_pos, self.another_object_rot, self.another_goal_pos, self.another_goal_rot, self.object_left_handle_pos, self.object_right_handle_pos, 
            self.left_hand_pos, self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, self.this_task
        )

        self.extras['total_successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # refresh
        tem_shadow_hand_dof = torch.zeros((self.num_envs, 48, 2), dtype=torch.float, device=self.device)
        pointer = 0
        for i in range(self.num_envs):
            tem_shadow_hand_dof[i, :, :] = self.dof_state[pointer:pointer+48, :]
            pointer += 48 + self.object_dof[i] * 2

        self.shadow_hand_dof_state = tem_shadow_hand_dof[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.shadow_hand_another_dof_state = tem_shadow_hand_dof[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2]
        self.shadow_hand_another_dof_pos = self.shadow_hand_another_dof_state[..., 0]
        self.shadow_hand_another_dof_vel = self.shadow_hand_another_dof_state[..., 1]

        self.num_total_rigid_body = self.total_rigid_body_tensor.shape[0]
        self.rigid_body_states = torch.zeros((self.num_envs, 52, 13), dtype=torch.float, device=self.device)
        pointer = 0
        for i in range(self.num_envs):
            self.rigid_body_states[i, :, :] = self.total_rigid_body_tensor[pointer:pointer+52, :]
            pointer += 52 + self.object_rigid_body[i] * 2 + 1

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.another_object_pose = self.root_state_tensor[self.another_object_indices, 0:7]
        self.another_object_pos = self.root_state_tensor[self.another_object_indices, 0:3]
        self.another_object_rot = self.root_state_tensor[self.another_object_indices, 3:7]
        self.another_object_linvel = self.root_state_tensor[self.another_object_indices, 7:10]
        self.another_object_angvel = self.root_state_tensor[self.another_object_indices, 10:13]

        self.object_left_handle_pos = self.rigid_body_states[:, 26 * 2 - 1, 0:3]
        self.object_left_handle_rot = self.rigid_body_states[:, 26 * 2 - 1, 3:7]
        self.object_left_handle_pos = self.object_left_handle_pos + quat_apply(self.object_left_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.5)
        self.object_left_handle_pos = self.object_left_handle_pos + quat_apply(self.object_left_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.39)
        self.object_left_handle_pos = self.object_left_handle_pos + quat_apply(self.object_left_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.04)

        self.object_right_handle_pos = self.rigid_body_states[:, 26 * 2 - 1, 0:3]
        self.object_right_handle_rot = self.rigid_body_states[:, 26 * 2 - 1, 3:7]
        self.object_right_handle_pos = self.object_right_handle_pos + quat_apply(self.object_right_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.5)
        self.object_right_handle_pos = self.object_right_handle_pos + quat_apply(self.object_right_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.39)
        self.object_right_handle_pos = self.object_right_handle_pos + quat_apply(self.object_right_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.04)

        self.left_hand_pos = self.rigid_body_states[:, 3 + 26, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, 3 + 26, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        self.right_hand_ff_pos = self.rigid_body_states[:, 7, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, 7, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, 11, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, 11, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, 15, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, 15, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, 20, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, 20, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, 25, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, 25, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.left_hand_ff_pos = self.rigid_body_states[:, 7 + 26, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, 7 + 26, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, 11 + 26, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, 11 + 26, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_rf_pos = self.rigid_body_states[:, 15 + 26, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, 15 + 26, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, 20 + 26, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, 20 + 26, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, 25 + 26, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, 25 + 26, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:13]
        self.fingertip_another_pos = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:3]
        
        self.compute_full_state()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        # fingertip observations, state(pose and vel) + force-torque sensors
        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)

        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 72  # 168 = 157 + 11
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        
        hand_pose_start = fingertip_obs_start + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.hand_positions[self.hand_indices, :]
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # another_hand
        another_hand_start = action_obs_start + 26
        self.obs_buf[:, another_hand_start:self.num_shadow_hand_dofs + another_hand_start] = unscale(self.shadow_hand_another_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs + another_hand_start:2*self.num_shadow_hand_dofs + another_hand_start] = self.vel_obs_scale * self.shadow_hand_another_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs + another_hand_start:3*self.num_shadow_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, 24:48]

        fingertip_another_obs_start = another_hand_start + 72
        self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states:fingertip_another_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, 30:]

        hand_another_pose_start = fingertip_another_obs_start + 95
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.hand_positions[self.another_hand_indices, :]
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 26] = self.actions[:, 26:]

        obj_obs_start = action_another_obs_start + 26  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        another_obj_obs_start = obj_obs_start + 13  # 157 = 144 + 13
        self.obs_buf[:, another_obj_obs_start:another_obj_obs_start + 7] = self.another_object_pose
        self.obs_buf[:, another_obj_obs_start + 7:another_obj_obs_start + 10] = self.another_object_linvel
        self.obs_buf[:, another_obj_obs_start + 10:another_obj_obs_start + 13] = self.vel_obs_scale * self.another_object_angvel
        # self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        # self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.another_goal_states[env_ids, 0:3] = self.another_goal_init_state[env_ids, 0:3]
        self.another_goal_states[env_ids, 3:7] = new_rot

        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])
        
        self.root_state_tensor[self.another_goal_object_indices[env_ids], 0:3] = self.another_goal_states[env_ids, 0:3]
        self.root_state_tensor[self.another_goal_object_indices[env_ids], 3:7] = self.another_goal_states[env_ids, 3:7]
        self.root_state_tensor[self.another_goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.another_goal_object_indices[env_ids], 7:13])

        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.another_object_indices[env_ids]] = self.another_object_init_state[env_ids].clone()

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])        
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.another_object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_another_dof_pos[env_ids, :] = pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]   

        self.shadow_hand_another_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        self.prev_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos
        self.cur_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                 another_hand_indices]).to(torch.int32))

        self.total_dof_target = torch.zeros(self.num_total_dof, dtype=torch.float, device=self.device)
        pointer = 0
        for i in range(self.num_envs):
            self.total_dof_target[pointer:pointer+48] = self.prev_targets[i]
            pointer += 48 + self.object_dof[i] * 2

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.total_dof_target),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        self.hand_positions[all_hand_indices.to(torch.long), :] = self.saved_root_tensor[all_hand_indices.to(torch.long), 0:3]
        self.hand_orientations[all_hand_indices.to(torch.long), :] = self.saved_root_tensor[all_hand_indices.to(torch.long), 3:7]
        all_indices = torch.unique(torch.cat([all_hand_indices,
                                                object_indices]).to(torch.int32))

        self.dof_state[:] = self.saved_dof_state.clone()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions[:, :20]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:26],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices + 24] = scale(self.actions[:, 32:52],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices + 24] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices + 24] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices + 24] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 24],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_forces[:, 1 + 26, :] = self.actions[:, 26:29] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            self.apply_torque[:, 1 + 26, :] = self.actions[:, 29:32] * self.dt * self.orientation_scale * 1000              

            self.total_apply_force = torch.zeros((self.num_total_rigid_body, 3), dtype=torch.float, device=self.device)
            self.total_apply_torque = torch.zeros((self.num_total_rigid_body, 3), dtype=torch.float, device=self.device)
            pointer = 0
            for i in range(self.num_envs):
                self.total_apply_force[pointer:pointer+52] = self.apply_forces[i]
                self.total_apply_torque[pointer:pointer+52] = self.apply_torque[i]
                pointer += 52 + self.object_rigid_body[i] * 2 + 1

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.total_apply_force), gymtorch.unwrap_tensor(self.total_apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_targets[:, self.actuated_dof_indices + 24] = self.cur_targets[:, self.actuated_dof_indices + 24]

        self.total_dof_target = torch.zeros(self.num_total_dof, dtype=torch.float, device=self.device)
        pointer = 0
        for i in range(self.num_envs):
            self.total_dof_target[pointer:pointer+48] = self.cur_targets[i]
            pointer += 48 + self.object_dof[i] * 2

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.total_dof_target))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

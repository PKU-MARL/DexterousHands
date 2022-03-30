from cgitb import reset
from cmath import pi
from copy import copy, deepcopy
import os
import math
from re import L, S
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import numpy as np
from random import shuffle
import torch.nn.functional as F
from gym.spaces.box import Box
from tasks.arm_base.multiAgent import Agent, MultiAgentEnv
import ipdb

class OneFranka(MultiAgentEnv) :

	def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

		self.cfg = cfg
		self.sim_params = sim_params
		self.physics_engine = physics_engine
		self.agent_index = agent_index
		self.is_multi_agent = is_multi_agent

		self.table_actor_list = []
		self.franka1_actor_list = []
		if not hasattr(self, "obs_dim") :
			self.obs_dim = 24 + 26
		
		super().__init__(cfg)

	def _load_franka(self, sim):
		asset_root = "./assets/"
		asset_file = "franka_description/robots/franka_panda.urdf"
		asset_options = gymapi.AssetOptions()
		asset_options.fix_base_link = True
		asset_options.disable_gravity = True
		# Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
		asset_options.flip_visual_attachments = True
		asset_options.armature = 0.01
		asset = self.gym.load_asset(sim, asset_root, asset_file, asset_options)
		return asset

	def _load_table(self, sim):
		table_dims = gymapi.Vec3(0.6, 0.8, 0.4)
		asset_options = gymapi.AssetOptions()
		asset_options.fix_base_link = True
		table_asset = self.gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
		return table_asset

	def _get_franka1_pos(self):
		franka_pose1 = gymapi.Transform()
		franka_pose1.p = gymapi.Vec3(0.35, -0.6, 0)  # 机械臂1的初始位置
		franka_pose1.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi/2)
		return franka_pose1

	def _get_table_pos(self):
		table_pose = gymapi.Transform()
		table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * 0.3)   #table的初始位置
		return table_pose

	def _get_dof_property(self, asset) :
		dof_props = self.gym.get_asset_dof_properties(asset)
		dof_num = self.gym.get_asset_dof_count(asset)
		dof_lower_limits = []
		dof_upper_limits = []
		dof_max_torque = []
		for i in range(dof_num) :
			dof_max_torque.append(dof_props['effort'][i])
			dof_lower_limits.append(dof_props['lower'][i])
			dof_upper_limits.append(dof_props['upper'][i])
		dof_max_torque = np.array(dof_max_torque)
		dof_lower_limits = np.array(dof_lower_limits)
		dof_upper_limits = np.array(dof_upper_limits)
		return dof_max_torque, dof_lower_limits, dof_upper_limits
	
	def _load_obj(self, env_ptr, env_id):
		obj_asset_list = []
		obj_pos_list = []
		for i, (asset, pos) in enumerate(zip(obj_asset_list, obj_pos_list)):
			obj_actor = self.gym.create_actor(env_ptr, asset, pos, "object{}".format(i), i, 1)
			self.actors.append(obj_actor)

	def _place_agents(self):
		spacing = self.cfg["env"]["envSpacing"]
		env_num = self.cfg["env"]["numEnvs"]
		lower = gymapi.Vec3(-spacing, -spacing, 0.0)
		upper = gymapi.Vec3(spacing, spacing, spacing)
		num_per_row = int(np.sqrt(env_num))
		self.env_num = env_num
		# self.ant_num = self.cfg["task"]["antNum"]

		# load asset
		self.franka_asset1 = self._load_franka(self.sim)
		self.table_asset = self._load_table(self.sim)
		self.franka1_dof_max_torque, self.franka1_dof_lower_limits, self.franka1_dof_upper_limits = self._get_dof_property(self.franka_asset1)

		dof_props = self.gym.get_asset_dof_properties(self.franka_asset1)
		# use position drive for all dofs
		if self.cfg["env"]["driveMode"] == "pos":
			dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
			dof_props["stiffness"][:7].fill(400.0)
			dof_props["damping"][:7].fill(40.0)
		else:	   # osc
			dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
			dof_props["stiffness"][:7].fill(0.0)
			dof_props["damping"][:7].fill(0.0)
		# grippers
		dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
		dof_props["stiffness"][7:].fill(800.0)
		dof_props["damping"][7:].fill(40.0)
		
		# set start pose
		franka_pose1 = self._get_franka1_pos()
		table_pose = self._get_table_pos()

		# set start dof
		franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset1)
		default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
		default_dof_pos[:7] = (self.franka1_dof_lower_limits + self.franka1_dof_upper_limits)[:7] * 0.3
		# grippers open
		default_dof_pos[7:] = self.franka1_dof_upper_limits[7:]

		franka1_dof_state = np.zeros_like(self.franka1_dof_max_torque, gymapi.DofState.dtype)
		franka1_dof_state["pos"] = default_dof_pos

		for env_id in range(env_num) :
			env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
			self.env_ptr_list.append(env_ptr)
			franka_actor1 = self.gym.create_actor(env_ptr, self.franka_asset1, franka_pose1, "franka1", env_id, 1)

			self.gym.set_actor_dof_properties(env_ptr, franka_actor1, dof_props)

			self.gym.set_actor_dof_states(env_ptr, franka_actor1, franka1_dof_state, gymapi.STATE_ALL)

			# print(self.gym.get_actor_dof_dict(env_ptr, franka_actor2))
			# hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, franka_actor2, "panda_hand", gymapi.DOMAIN_ENV)

			# print(self.gym.get_actor_rigid_body_dict(env_ptr, franka_actor1))
			
			self._load_obj(env_ptr, env_id)
		
			# box_actor = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, 0)
			# color = gymapi.Vec3(1, 0, 0)
			# self.gym.set_rigid_body_color(env_ptr, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
			table_actor = self.gym.create_actor(env_ptr, self.table_asset, table_pose, "table", env_id, 1)

			self.table_actor_list.append(table_actor)
			self.franka1_actor_list.append(franka_actor1)
			self.actors.append(franka_actor1)
			self.actors.append(table_actor)

	def _create_ground_plane(self) :
		plane_params = gymapi.PlaneParams()
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		plane_params.static_friction = 0.1
		plane_params.dynamic_friction = 0.1
		self.gym.add_ground(self.sim, plane_params)
	
	def _get_reward_done(self) :
		reward = torch.zeros((self.env_num,), device=self.device)  #  16
		return reward, reward
		
	def _refresh_observation(self) :
		
		self.obs_buffer[:, :self.dof_dim*2].copy_(self.dof_state_tensor.view(self.env_num, -1))
		self.obs_buffer[:,self.dof_dim*2:].copy_(self.root_tensor.view(self.env_num, -1))
	
	def _perform_actions(self):
		pos_act = torch.zeros_like(self.act_buffer)
		eff_act = torch.zeros_like(self.act_buffer)
		if self.cfg["env"]["driveMode"] == "pos" :
			pos_act[:, :7] = self.act_buffer[:, :7]
		else :
			eff_act[:, :7] = self.act_buffer[:, :7]
		pos_act[:, 7:9] = self.act_buffer[:, 7:9]
		self.gym.set_dof_position_target_tensor(
			self.sim, gymtorch.unwrap_tensor(pos_act.view(-1))
		)
		self.gym.set_dof_actuation_force_tensor(
			self.sim, gymtorch.unwrap_tensor(eff_act.view(-1))
		)

	def step(self, actions) :

		self.act_buffer.copy_(actions)
		self._perform_actions()
		
		self.gym.simulate(self.sim)
		self.gym.fetch_results(self.sim, True)
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)

		reward, done = self._get_reward_done()
		self._refresh_observation()

		if not self.headless :
			self.render()

		return self.obs_buffer, reward, done, "No info presented"

	def reset(self, to_reset = "all") :

		"""
		reset those need to be reseted
		"""

		if to_reset == "all" :
			to_reset = np.ones((self.env_num,))
		reseted = False
		for env_id,reset in enumerate(to_reset) :
			if reset.item() :
				self.dof_state_tensor[env_id].copy_(self.initial_dof_states[env_id])
				self.root_tensor[env_id].copy_(self.initial_root_states[env_id])
				reseted = True
				self.episode_buffer[env_id] = 0
				self.reset_buffer[env_id] = 0
		
		if reseted :
			self.gym.set_dof_state_tensor(
				self.sim,
				gymtorch.unwrap_tensor(self.dof_state_tensor)
			)
			self.gym.set_actor_root_state_tensor(
				self.sim,
				gymtorch.unwrap_tensor(self.root_tensor)
			)
		
		reward, done = self._get_reward_done()
		torch.fill_(self.obs_buffer, 0)
		self._refresh_observation()

		return self.obs_buffer, reward, done, "No info presented"
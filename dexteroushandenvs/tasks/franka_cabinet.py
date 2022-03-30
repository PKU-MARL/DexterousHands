from cgitb import reset
from cmath import pi
from copy import copy, deepcopy
import enum
import os
import math
from re import L, S
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import numpy as np
from random import shuffle
from torch import device, nonzero, rand
import torch.nn.functional as F
from gym.spaces.box import Box
from tasks.arm_base.oneFranka import OneFranka
from tasks.hand_base.base_task import BaseTask
from utils.contact_buffer import ContactBuffer
from tqdm import tqdm
import ipdb

class OneFrankaCabinet(BaseTask) :

	def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

		self.cfg = cfg
		self.sim_params = sim_params
		self.physics_engine = physics_engine
		self.agent_index = agent_index
		self.is_multi_agent = is_multi_agent
		self.up_axis = 'z'
		self.cfg["device_type"] = device_type
		self.cfg["device_id"] = device_id
		self.cfg["headless"] = headless
		self.device_type = device_type
		self.device_id = device_id
		self.headless = headless
		self.device = "cpu"
		if self.device_type == "cuda" or self.device_type == "GPU":
			self.device = "cuda" + ":" + str(self.device_id)
		self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
		
		self.env_num = cfg["env"]["numEnvs"]
		self.asset_root = cfg["env"]["asset"]["assetRoot"]
		self.cabinet_num = cfg["env"]["asset"]["cabinetAssetNum"]
		cabinet_list_len = len(cfg["env"]["asset"]["cabinetAssets"])
		self.cabinet_name_list = []
		print("Simulator: number of cabinets:", self.cabinet_num)
		assert(self.cabinet_num <= cabinet_list_len)	# the number of used length must less than real length
		assert(self.env_num % self.cabinet_num == 0)	# each cabinet should have equal number envs
		self.env_per_cabinet = self.env_num // self.cabinet_num
		for name in cfg["env"]["asset"]["cabinetAssets"] :
			self.cabinet_name_list.append(cfg["env"]["asset"]["cabinetAssets"][name]["name"])
		self.cabinet_dof_lower_limits_tensor = torch.zeros((self.cabinet_num, 1), device=self.device)
		self.cabinet_dof_upper_limits_tensor = torch.zeros((self.cabinet_num, 1), device=self.device)

		self.env_ptr_list = []
		self.obj_loaded = False
		self.franka_loaded = False

		super().__init__(cfg=self.cfg)

		# acquire tensors
		self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
		self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
		self.rigid_body_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))

		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)

		self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
		self.dof_state_tensor = self.dof_state_tensor.view(self.num_envs, -1, 2)
		self.rigid_body_tensor = self.rigid_body_tensor.view(self.num_envs, -1, 13)

		self.initial_dof_states = self.dof_state_tensor.clone()
		self.initial_root_states = self.root_tensor.clone()

		# precise slices of tensors
		env_ptr = self.env_ptr_list[0]
		franka1_actor = self.franka_actor_list[0]
		cabinet_actor = self.cabinet_actor_list[0]
		self.hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
			env_ptr,
			franka1_actor,
			"panda_hand",
			gymapi.DOMAIN_ENV
		)
		self.cabinet_rigid_body_index = self.gym.find_actor_rigid_body_index(
			env_ptr,
			cabinet_actor,
			self.cabinet_rig_name,
			gymapi.DOMAIN_ENV
		)
		self.cabinet_dof_index = self.gym.find_actor_dof_index(
			env_ptr,
			cabinet_actor,
			self.cabinet_dof_name,
			gymapi.DOMAIN_ENV
		)
		self.hand_rigid_body_tensor = self.rigid_body_tensor[:, self.hand_rigid_body_index, :]
		self.cabinet_dof_tensor = self.dof_state_tensor[:, self.cabinet_dof_index, :]
		self.cabinet_dof_tensor_spec = self.cabinet_dof_tensor.view(self.cabinet_num, self.env_per_cabinet, -1)
		self.cabinet_door_rigid_body_tensor = self.rigid_body_tensor[:, self.cabinet_rigid_body_index, :]
		self.dof_dim = 10
		self.pos_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
		self.eff_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
		if cfg["task"]["target"] == "close" :
			self.initial_dof_states.view(self.cabinet_num, self.env_per_cabinet, -1, 2)[:, :, self.cabinet_dof_index, 0] = ((self.cabinet_dof_upper_limits_tensor[:, 0] + self.cabinet_dof_lower_limits_tensor[:, 0]) / 2).view(self.cabinet_num, -1)
			self.success_dof_states = self.cabinet_dof_lower_limits_tensor[:, 0].clone()
		else :
			self.initial_dof_states.view(self.cabinet_num, self.env_per_cabinet, -1, 2)[:, :, self.cabinet_dof_index, 0] = self.cabinet_dof_lower_limits_tensor[:, 0].view(self.cabinet_num, -1)
			self.success_dof_states = self.cabinet_dof_upper_limits_tensor[:, 0].clone()

		if cfg["task"]["target"] == "close" :
			self.cabinet_dof_coef = -1.0
		else :
			self.cabinet_dof_coef = +1.0
		
		# init collision buffer
		self.contact_buffer_size = cfg["env"]["contactBufferSize"]
		self.contact_moving_threshold = cfg["env"]["contactMovingThreshold"]
		self.contact_save_path = cfg["env"]["contactSavePath"]
		self.contact_save_steps = cfg["env"]["contactSaveSteps"]
		self.contact_steps = 0
		self.contact_buffer_list = []
		assert(self.contact_buffer_size >= self.env_num)
		for i in range(self.cabinet_num) :
			self.contact_buffer_list.append(ContactBuffer(self.contact_buffer_size, 7, device=self.device))
		
		# params of randomization
		self.cabinet_reset_position_noise = cfg["env"]["reset"]["cabinet"]["resetPositionNoise"]
		self.cabinet_reset_rotation_noise = cfg["env"]["reset"]["cabinet"]["resetRotationNoise"]
		self.cabinet_reset_dof_pos_interval = cfg["env"]["reset"]["cabinet"]["resetDofPosRandomInterval"]
		self.cabinet_reset_dof_vel_interval = cfg["env"]["reset"]["cabinet"]["resetDofVelRandomInterval"]
		self.franka_reset_position_noise = cfg["env"]["reset"]["franka"]["resetPositionNoise"]
		self.franka_reset_rotation_noise = cfg["env"]["reset"]["franka"]["resetRotationNoise"]
		self.franka_reset_dof_pos_interval = cfg["env"]["reset"]["franka"]["resetDofPosRandomInterval"]
		self.franka_reset_dof_vel_interval = cfg["env"]["reset"]["franka"]["resetDofVelRandomInterval"]

		# params for success rate
		self.success = 0.0
		self.fail = 0.0

	def create_sim(self):
		self.dt = self.sim_params.dt
		self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

		self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
		self._create_ground_plane()
		self._place_agents(self.cfg["env"]["numEnvs"], self.cfg["env"]["envSpacing"])

	def _load_franka(self, env_ptr, env_id):

		if self.franka_loaded == False :

			self.franka_actor_list = []

			asset_root = self.asset_root
			asset_file = "franka_description/robots/franka_panda.urdf"
			asset_options = gymapi.AssetOptions()
			asset_options.fix_base_link = True
			asset_options.disable_gravity = True
			# Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
			asset_options.flip_visual_attachments = True
			asset_options.armature = 0.01
			self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

			self.franka_loaded = True

		franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits = self._get_dof_property(self.franka_asset)
		self.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=self.device)
		self.franka_dof_mean_limits_tensor = torch.tensor((franka_dof_lower_limits + franka_dof_upper_limits)/2, device=self.device)
		self.franka_dof_limits_range_tensor = torch.tensor((franka_dof_upper_limits - franka_dof_lower_limits)/2, device=self.device)
		self.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=self.device)
		self.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=self.device)

		dof_props = self.gym.get_asset_dof_properties(self.franka_asset)

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

		# root pose
		franka_pose = gymapi.Transform()
		franka_pose.p = gymapi.Vec3(1.1, 0.0, 0.5)
		franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

		# set start dof
		franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
		default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
		default_dof_pos[:7] = (franka_dof_lower_limits + franka_dof_upper_limits)[:7] * 0.3
		# grippers open
		default_dof_pos[7:] = franka_dof_upper_limits[7:]
		franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype)
		franka_dof_state["pos"] = default_dof_pos

		franka_actor = self.gym.create_actor(
			env_ptr,
			self.franka_asset, 
			franka_pose,
			"franka",
			env_id,
			2,
			0)

		self.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
		self.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
		self.franka_actor_list.append(franka_actor)

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

		if self.obj_loaded == False :

			self.cabinet_asset_list = []
			self.cabinet_pose_list = []
			self.cabinet_actor_list = []

			total_len = len(self.cfg["env"]["asset"]["cabinetAssets"].items())
			total_len = min(total_len, self.cabinet_num)

			with tqdm(total=total_len) as pbar:
				pbar.set_description('Loading cabinet assets:')

				for id, (name, val) in enumerate(self.cfg["env"]["asset"]["cabinetAssets"].items()) :

					if id >= self.cabinet_num :
						break

					asset_options = gymapi.AssetOptions()
					asset_options.fix_base_link = True
					asset_options.disable_gravity = True
					asset_options.collapse_fixed_joints = True
					cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, val["path"], asset_options)
					self.cabinet_asset_list.append(cabinet_asset)

					dof_dict = self.gym.get_asset_dof_dict(cabinet_asset)
					if len(dof_dict) != 1 :
						print(val["path"])
						print(len(dof_dict))
					assert(len(dof_dict) == 1)
					self.cabinet_dof_name = list(dof_dict.keys())[0]

					rig_dict = self.gym.get_asset_rigid_body_dict(cabinet_asset)
					assert(len(rig_dict) == 2)
					self.cabinet_rig_name = list(rig_dict.keys())[1]
					assert(self.cabinet_rig_name != "base")

					cabinet_start_pose = gymapi.Transform()
					cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(1.5, self.up_axis_idx))
					cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
					self.cabinet_pose_list.append(cabinet_start_pose)

					max_torque, lower_limits, upper_limits = self._get_dof_property(cabinet_asset)
					self.cabinet_dof_lower_limits_tensor[id, :] = torch.tensor(lower_limits[0], device=self.device)
					self.cabinet_dof_upper_limits_tensor[id, :] = torch.tensor(upper_limits[0], device=self.device)

					pbar.update(id + 1)

			self.obj_loaded = True

		cabinet_type = env_id // self.env_per_cabinet
		subenv_id = env_id % self.env_per_cabinet
		obj_actor = self.gym.create_actor(
			env_ptr,
			self.cabinet_asset_list[cabinet_type],
			self.cabinet_pose_list[cabinet_type],
			"cabinet{}-{}".format(cabinet_type, subenv_id),
			env_id,
			1,
			0)
		cabinet_dof_props = self.gym.get_asset_dof_properties(self.cabinet_asset_list[cabinet_type])
		cabinet_dof_props['damping'][0] = 10.0
		cabinet_dof_props["driveMode"][0] = gymapi.DOF_MODE_NONE
		self.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)
		self.cabinet_actor_list.append(obj_actor)

	def _place_agents(self, env_num, spacing):

		print("Simulator: creating agents")

		lower = gymapi.Vec3(-spacing, -spacing, 0.0)
		upper = gymapi.Vec3(spacing, spacing, spacing)
		num_per_row = int(np.sqrt(env_num))

		with tqdm(total=env_num) as pbar:
			pbar.set_description('Enumerating envs:')
			for env_id in range(env_num) :
				env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
				self.env_ptr_list.append(env_ptr)
				self._load_franka(env_ptr, env_id)
				self._load_obj(env_ptr, env_id)
				pbar.update(env_id + 1)
		

	def _create_ground_plane(self) :
		plane_params = gymapi.PlaneParams()
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		plane_params.static_friction = 0.1
		plane_params.dynamic_friction = 0.1
		self.gym.add_ground(self.sim, plane_params)
	
	def _get_reward_done(self) :

		self.rew_buf = self.cabinet_dof_coef * self.cabinet_dof_tensor[:, 0]
		diff_from_success = torch.abs(self.cabinet_dof_tensor_spec[:, :, 0]-self.success_dof_states.view(self.cabinet_num, -1)).view(-1)
		success = (diff_from_success < 0.1)
		fail = (self.progress_buf >= self.max_episode_length)
		self.reset_buf = (self.reset_buf | success | fail)
		new_success = success.sum().detach().cpu().item()
		new_fail = fail.sum().detach().cpu().item()
		self.success += new_success
		self.fail += new_fail
		if (self.success + self.fail) > 1000 :
			self.success *= 0.5
			self.fail *= 0.5
		return self.rew_buf, self.reset_buf
		
	def _refresh_observation(self) :

		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)

		p = self.dof_dim*2
		q = p + 13*2
		self.obs_buf[:,:p].copy_(self.dof_state_tensor.view(self.env_num, -1))
		self.obs_buf[:,p:q].copy_(self.root_tensor.view(self.env_num, -1))
		self.obs_buf[:,q:].copy_(self.hand_rigid_body_tensor.view(self.env_num, -1))
	
	def _perform_actions(self, actions):

		actions = actions.to(self.device)

		if self.cfg["env"]["driveMode"] == "pos" :
			self.pos_act[:, :7] = actions[:, :7] * self.franka_dof_limits_range_tensor[:7] + self.franka_dof_mean_limits_tensor[:7]
		else :
			self.eff_act[:, :7] = actions[:, :7] * self.franka_dof_max_torque_tensor[:7]
		self.pos_act[:, 7:9] = actions[:, 7:9] * self.franka_dof_limits_range_tensor[7:9] + self.franka_dof_mean_limits_tensor[7:9]
		self.gym.set_dof_position_target_tensor(
			self.sim, gymtorch.unwrap_tensor(self.pos_act.view(-1))
		)
		self.gym.set_dof_actuation_force_tensor(
			self.sim, gymtorch.unwrap_tensor(self.eff_act.view(-1))
		)
	
	def _refresh_contact(self) :

		door_pos = self.cabinet_door_rigid_body_tensor[:, :3]
		door_rot = self.cabinet_door_rigid_body_tensor[:, 3:7]
		hand_pos = self.hand_rigid_body_tensor[:, :3]
		hand_rot = self.hand_rigid_body_tensor[:, 3:7]
		cabinet_dof_spd = torch.abs(self.cabinet_dof_tensor[:, 1])
		moving_mask = (cabinet_dof_spd > self.contact_moving_threshold).view(self.cabinet_num, self.env_per_cabinet)

		relative_pos = quat_apply(quat_conjugate(door_rot), hand_pos - door_pos)
		relative_rot = quat_mul(quat_conjugate(door_rot), hand_rot)				# is there any bug?
		pose = torch.cat((relative_pos, relative_rot), dim=1).view(self.cabinet_num, self.env_per_cabinet, -1)
		
		for i, buffer in enumerate(self.contact_buffer_list) :
			non_zero_idx = torch.nonzero(moving_mask[i], as_tuple=True)[0]
			buffer.insert(pose[i, non_zero_idx])
		
		self.contact_steps += 1

		if self.contact_steps % self.contact_save_steps == 0 :
			for buffer, name in zip(self.contact_buffer_list, self.cabinet_name_list) :
				dir = os.path.join(self.contact_save_path, name)
				if not os.path.exists(dir) :
					os.makedirs(dir)
				buffer.save(os.path.join(dir, str(self.contact_steps) + ".pt"))


	def step(self, actions) :

		self._perform_actions(actions)
		
		self.gym.simulate(self.sim)
		self.gym.fetch_results(self.sim, True)

		self.progress_buf += 1

		self._refresh_observation()
		reward, done = self._get_reward_done()
		self._refresh_contact()

		if not self.headless :
			self.render()

		# print("step", self.obs_buf)
		self.reset(self.reset_buf)

		success_rate = self.success / (self.fail + self.success + 1e-8)
		print("success_rate:", success_rate)
		
		return self.obs_buf, self.rew_buf, self.reset_buf, success_rate
	
	def reset(self, to_reset = "all") :

		"""
		reset those need to be reseted
		"""

		if to_reset == "all" :
			to_reset = np.ones((self.env_num,))
		reseted = False
		for env_id,reset in enumerate(to_reset) :
			if reset.item() :
				# need randomization
				reset_dof_states = self.initial_dof_states[env_id].clone()
				reset_root_states = self.initial_root_states[env_id].clone()
				franka_reset_pos_tensor = reset_root_states[0, :3]
				franka_reset_rot_tensor = reset_root_states[0, 3:7]
				franka_reset_dof_pos_tensor = reset_dof_states[:9, 0]
				franka_reset_dof_vel_tensor = reset_dof_states[:9, 1]
				cabinet_reset_pos_tensor = reset_root_states[1, :3]
				cabinet_reset_rot_tensor = reset_root_states[1, 3:7]
				cabinet_reset_dof_pos_tensor = reset_dof_states[9:, 0]
				cabinet_reset_dof_vel_tensor = reset_dof_states[9:, 1]

				cabinet_type = env_id // self.env_per_cabinet
				self.intervaledRandom_(franka_reset_pos_tensor, self.franka_reset_position_noise)
				self.intervaledRandom_(franka_reset_rot_tensor, self.franka_reset_rotation_noise)
				self.intervaledRandom_(franka_reset_dof_pos_tensor, self.franka_reset_dof_pos_interval, self.franka_dof_lower_limits_tensor, self.franka_dof_upper_limits_tensor)
				self.intervaledRandom_(franka_reset_dof_vel_tensor, self.franka_reset_dof_vel_interval)
				self.intervaledRandom_(cabinet_reset_pos_tensor, self.cabinet_reset_position_noise)
				self.intervaledRandom_(cabinet_reset_rot_tensor, self.cabinet_reset_rotation_noise)
				self.intervaledRandom_(cabinet_reset_dof_pos_tensor, self.cabinet_reset_dof_pos_interval, self.cabinet_dof_lower_limits_tensor[cabinet_type], self.cabinet_dof_upper_limits_tensor[cabinet_type])
				self.intervaledRandom_(cabinet_reset_dof_vel_tensor, self.cabinet_reset_dof_vel_interval)

				self.dof_state_tensor[env_id].copy_(reset_dof_states)
				self.root_tensor[env_id].copy_(reset_root_states)
				reseted = True
				self.progress_buf[env_id] = 0
				self.reset_buf[env_id] = 0
		
		if reseted :
			self.gym.set_dof_state_tensor(
				self.sim,
				gymtorch.unwrap_tensor(self.dof_state_tensor)
			)
			self.gym.set_actor_root_state_tensor(
				self.sim,
				gymtorch.unwrap_tensor(self.root_tensor)
			)
			
		self.gym.simulate(self.sim)
		self.gym.fetch_results(self.sim, True)
		self._refresh_observation()
		reward, done = self._get_reward_done()

		success_rate = self.success / (self.fail + self.success + 1e-8)
		return self.obs_buf, self.rew_buf, self.reset_buf, success_rate
	
	def intervaledRandom_(self, tensor, dist, lower=None, upper=None) :
		tensor += torch.rand(tensor.shape, device=self.device)*dist*2 - dist
		if lower is not None and upper is not None :
			torch.clamp_(tensor, min=lower, max=upper)
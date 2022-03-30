from abc import ABC, abstractclassmethod, abstractmethod
from cgitb import reset
import math
from re import M
import gym
import numpy as np
import os
from pyparsing import col
import shutil
import torch
import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple
from tensorboardX import SummaryWriter
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.arm_base.vec_task import VecTask

from abc import ABC, abstractmethod

class MultiAgentEnv(VecTask) :

	def __init__(self, cfg, sim_device, graphics_device_id, headless) :

		super().__init__(cfg, sim_device, graphics_device_id, headless)

		self.env_num = cfg["env"]["numEnvs"]
		if not hasattr(self, "obs_dim") :
			self.obs_dim = 0

		# setting up lists
		self.actors = []		# actors are described by urdf
		self.env_ptr_list = []

		self._build_sim()


	def _build_sim(self) :

		print("simulator: building sim")

		self.dt = self.sim_params.dt
		self.sim_params.up_axis = gymapi.UP_AXIS_Z
		self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
		self.sim_params.gravity.x = 0
		self.sim_params.gravity.y = 0
		self.sim_params.gravity.z = -9.81

		self.sim = super().create_sim(
			self.device_id,
			self.graphics_device_id,
			self.physics_engine,
			self.sim_params
		)

		print("simulator: bulding sim(1/5): vec-env created")

		self._create_ground_plane()

		print("simulator: bulding sim(2/5): ground plane created")

		self._place_agents()

		print("simulator: bulding sim(3/5): agents created")

		self.gym.prepare_sim(self.sim)
		self.sim_initialized = True

		# obtaining root tensors
		self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
		self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
		self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

		self.root_tensor = gymtorch.wrap_tensor(self.root_tensor)
		self.dof_state_tensor = gymtorch.wrap_tensor(self.dof_state_tensor)
		self.sensor_tensor = gymtorch.wrap_tensor(self.sensor_tensor)
		self.rigid_body_tensor = gymtorch.wrap_tensor(self.rigid_body_tensor)

		if self.root_tensor != None :
			self.root_tensor = self.root_tensor.view(self.env_num, -1, 13)
		if self.dof_state_tensor != None :
			self.dof_state_tensor = self.dof_state_tensor.view(self.env_num, -1, 2)
		if self.sensor_tensor != None :
			self.sensor_tensor = self.sensor_tensor.view(self.env_num, -1)
		if self.rigid_body_tensor != None :
			self.rigid_body_tensor = self.rigid_body_tensor.view(self.env_num, -1, 13)

		# refresh tensors
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_force_sensor_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)

		# collecting initial tensors
		self.initial_dof_states = self.dof_state_tensor.clone()
		self.initial_root_states = self.root_tensor.clone()

		# some dimensions
		self.dof_dim = self.dof_state_tensor.shape[1]
		self.rig_dim = self.rigid_body_tensor.shape[1]

		# buffers
		self.act_buffer = torch.zeros((self.env_num, self.dof_dim), device=self.device)
		self.obs_buffer = torch.zeros((self.env_num, self.obs_dim), device=self.device)	# size of obs is depend on environment
		self.reset_buffer = torch.zeros(self.env_num, device=self.device).long()
		self.episode_buffer = torch.zeros(self.env_num, device=self.device).long()

		print("simulator: bulding sim(4/5): buffers and tensors allocated")

		super().set_viewer()

		print("simulator: bulding sim(5/5): viewer set")

	def _perform_actions(self) :

		# set_dof_position_target_tensor
		# set_dof_actuation_force_tensor
		if self.cfg["env"]["driveMode"] == "pos" :
			self.gym.set_dof_position_target_tensor(
				self.sim, gymtorch.unwrap_tensor(self.act_buffer)
			)
		else :
			self.gym.set_dof_actuation_force_tensor(
				self.sim, gymtorch.unwrap_tensor(self.act_buffer)
			)

	def _create_ground_plane(self) :

		plane_params = gymapi.PlaneParams()
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.add_ground(self.sim, plane_params)
	
	@abstractmethod
	def _place_agents(self) :
		"""
		agent placement, need to be finished by subclasses
		"""
		return None

	@abstractmethod
	def _get_reward_done(self) :
		"""
		get current reward, need to be finished by subclasses
		"""
		return None
	
	@abstractmethod
	def step(self, actions) :
		"""
		get current reward, need to be finished by subclasses
		same as gym
		"""
		pass

	@abstractmethod
	def reset(self, reset_idx) :
		"""
		get current reward, need to be finished by subclasses
		same as gym
		"""
		pass

class Agent() :

	def __init__(self,
		cfg				# config file
	) :

		self.cfg = cfg
		self.name = cfg["name"]
		self.device = cfg["device"]
		
		# params need to be initialized
		self.asset = None
		self.algo = None
		self.obs_buffer = None
		self.act_buffer = None
		self.dof_buffer = None
		self.root_buffer = None
		self.reset_buffer = None
		self.ini_dof_state = None
		self.ini_root_state = None

		self.env_ptr_list = []
		self.col_id_list = []
		self.handle_list = []

	def set_act(self, act_buffer) :

		self.act_buffer = act_buffer

	def set_obs(self, obs_buffer) :

		self.obs_buffer = obs_buffer
	
	def set_dof(self, dof_buffer) :

		self.dof_buffer = dof_buffer
	
	def set_root(self, root_buffer) :

		self.root_buffer = root_buffer
	
	def set_ini_root(self, ini_root) :

		self.ini_root_state = ini_root
	
	def set_ini_dof(self, ini_dof) :

		self.ini_dof_state = ini_dof
	
	def set_reset(self, reset_buf) :

		self.reset_buffer = reset_buf
	
	def observe(self) :

		return self.obs_buffer
	
	def act(self, action) :

		if self.act_buffer != None :
			self.act_buffer.copy_(action)
	
	def set_pose(self, env_idx, root = None, pose = None) :
		if root == None :
			root = self.ini_root_state[env_idx]
		if pose == None and self.ini_dof_state != None:
			pose = self.ini_dof_state[env_idx]
		self.root_buffer[env_idx].copy_(root)
		if self.dof_buffer != None :
			self.dof_buffer[env_idx].copy_(pose)
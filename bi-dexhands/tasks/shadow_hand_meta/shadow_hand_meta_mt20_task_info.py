from asyncio import tasks
import this
from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch

from utils.torch_jit_utils import *
from tasks.hand_base.base_task import BaseTask

from isaacgym import gymtorch
from isaacgym import gymapi

def obtain_task_info(task_name):
    if task_name == "hand_over":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [ 0, -0.64, 0.54, 0, -0., 0.]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_underarm":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [ 0, -0.79, 0.54, 0, -0., 0.]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_over2underarm":
        hand_start_pose = [0, 0, 0.5, 1.57 + 0.1, 3.14, 0]
        another_hand_start_pose = [0, -0.8, 0.5, 0, 0, 3.1415]
        object_pose = [0, 0.05, 1.0, 0, 0, 0]
        goal_pose = [0, -0.40, 0.55, 0, -0., 0.]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "two_catch_underarm":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        another_object_pose = [0, -0.79, 0.54, 0, 0, 0]
        goal_pose = [0, -0.79, 0.54, -0., -0., 0.]
        another_goal_pose = [-0.05, -0.39, 0.54, -0., -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_abreast":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
        object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
        goal_pose = [-0.39, -1.15, 0.54, 0, 0, 0]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "push_block":
        hand_start_pose = [0.55, 0.2, 0.8, 3.14159, 0, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.8, 3.14159, 0, 1.57]
        object_pose = [0.0, 0.2, 0.6, 1.57, 1.57, 0]
        goal_pose = [-0.3, 0, 0.6, 0, 0, 0]
        another_object_pose = [0.0, -0.2, 0.6, 1.57, 1.57, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]
        table_pose_dim = [0, 0, 0.3, 0, 0, 0, 1.0, 1.0, 0.6]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "block"

    if task_name == "block_stack":
        hand_start_pose = [0.0, 0.6, 0.7, 3.14159, 0, 3.14]
        another_hand_start_pose = [0.0, -0.6, 0.7, 3.14159, 0, 0]
        object_pose = [0.0, 0.1, 0.625, 1.57, 1.57, 0]
        goal_pose = [-0.39, -1.15, 0.54, 0, 0, 0]
        another_object_pose = [0.0, -0.1, 0.625, 1.57, 1.57, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]
        table_pose_dim = [0, 0, 0.3, 0, 0, 0, 0.5, 1.0, 0.6]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 100
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "block"

    if task_name == "re_orientation":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0.0, -0.39, 0.60, 0, 0, 0]
        goal_pose = [0.0, -0.39, 0.60, 0, 0, 0]
        another_object_pose = [0.0, -0.76, 0.10, 0, 0, 0]
        another_goal_pose = [0.0, -0.76, 0.10, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "block"

    if task_name == "grasp_and_place":
        hand_start_pose = [0.55, 0.2, 0.8, 3.14159, 0, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.8, 3.14159, 0, 1.57]
        object_pose = [0.0, 0.2, 0.6, 0, 0, 0]
        goal_pose = [-0.39, -1.15, 0.54, 0, 0, 0]
        another_object_pose = [0.0, -0.2, 0.6, 1.57, 1.57, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]
        table_pose_dim = [0, 0, 0.3, 0, 0, 0, 1.0, 1.0, 0.6]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "block"

    if task_name == "door_open_outward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 0.0, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]   
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "door"

    if task_name == "door_close_inward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 3.14159, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]   
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "door"

    if task_name == "door_open_inward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 3.14159, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]   
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "door"

    if task_name == "door_close_outward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 0.0, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]   
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "door"

    if task_name == "switch":
        hand_start_pose = [0.55, 0.2, 0.8, 3.14159, 0, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.8, 3.14159, 0, 1.57]
        object_pose = [0.0, 0.2, 0.65, 3.141592, 1.57, 0]
        goal_pose = [0, 0.0, 10, 0, 0, 0]
        another_object_pose = [0.0, -0.2, 0.65, 3.141592, 1.57, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, 0, 0.275, 0, 0, 0, 0.5, 1.0, 0.55]

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        # object_asset_options.collapse_fixed_joints = True
        # object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        # object_asset_options.replace_cylinder_with_capsule = True
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        object_type = "switch"

    if task_name == "lift_underarm":
        hand_start_pose = [0, 0.05, 0.45, 0, 0, 0]
        another_hand_start_pose = [0, -1.25, 0.45, 0, 0, 3.14159]
        object_pose = [0, -0.6, 0.45, 0, 0.0, 0.0]
        goal_pose = [0, 0.0, 1, 0, 0, 0]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, -0.6, 0.2, 0, 0, 0, 0.3, 0.3, 0.4]

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 1000
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "pot"

    if task_name == "pen":
        hand_start_pose = [0.55, 0.2, 0.8, 3.14159, 0, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.8, 3.14159, 0, 1.57]
        object_pose = [0.0, 0., 0.6, 1.57, 1.57, 0]
        goal_pose = [0, 0.0, 10, 0, 0, 0]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, 0, 0.3, 0, 0, 0, 0.5, 1.0, 0.6]

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "pen"

    if task_name == "swing_cup":
        hand_start_pose = [0.35, 0.2, 0.8, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.35, -0.2, 0.8, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.6, 0, 0, 1.57]
        goal_pose = [0, 0.0, 10, 0, 0, 0]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, 0.0, 0.3, 0, 0, 0, 0.3, 0.3, 0.6]

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "cup"

    if task_name == "scissors":
        hand_start_pose = [0.55, 0.2, 0.8, 3.14159, 0, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.8, 3.14159, 0, 1.57]
        object_pose = [0.0, 0., 0.6, 0, 0, 1.57]
        goal_pose = [0, 0.0, 10, 0, 0, 0]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, 0.0, 0.3, 0, 0, 0, 0.5, 1.0, 0.6]

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "scissors"

    if task_name == "bottle_cap":
        hand_start_pose = [0, -1.05, 0.5, 0.5, 3.14159, 3.14159]
        another_hand_start_pose = [0, -0.25, 0.45, 0, 1.57 - 0.7, 0]
        object_pose = [0, -0.6, 0.5, 0, -0.7, 0]
        goal_pose = [0, 0.0, 5, 0, 0, 0]
        another_object_pose = [0, 0, 10, 0, 0, 0]
        another_goal_pose = [0, 0, 10, 0, 0, 0]             
        table_pose_dim = [0.0, -0.6, 0.05, 0, 0, 0, 0.3, 0.3, 0.1]

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "bottle"

    return hand_start_pose, another_hand_start_pose, object_pose, goal_pose, another_object_pose, another_goal_pose, table_pose_dim, object_asset_options, object_type

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, another_object_pos, another_object_rot, another_target_pos, another_target_rot, object_left_handle_pos, object_right_handle_pos,
    left_hand_pos, right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, this_task: str
):
    # Distance from the hand to the object
    if this_task in ["catch_underarm", "two_catch_underarm", "catch_abreast", "catch_over2underarm", "hand_over"]:
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        another_goal_dist = torch.norm(another_target_pos - another_object_pos, p=2, dim=-1)

        # Orientation alignment for the cube in hand and goal cube
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        another_quat_diff = quat_mul(another_object_rot, quat_conjugate(another_target_rot))
        another_rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(another_quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        dist_rew = goal_dist
        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = torch.exp(-0.2*((dist_rew * dist_reward_scale + rot_dist) + (another_goal_dist * dist_reward_scale)))

        # Find out which envs hit the goal and update successes count
        goal_resets = torch.where(torch.abs(goal_dist) <= 0.03, torch.ones_like(reset_goal_buf), reset_goal_buf)
        successes = successes + goal_resets

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

        # Fall penalty: distance to the goal is larger than a threashold
        reward = torch.where(object_pos[:, 2] <= 0.1, reward + fall_penalty, reward)
        reward = torch.where(another_object_pos[:, 2] <= 0.1, reward + fall_penalty, reward)

        # Check env termination conditions, including maximum success number
        resets = torch.where(object_pos[:, 2] <= 0.1, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(another_object_pos[:, 2] <= 0.1, torch.ones_like(resets), resets)        
        
        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        # Apply penalty for not reaching the goal
        if max_consecutive_successes > 0:
            reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)
        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["door_open_inward"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        

        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.5,
                        torch.where(left_hand_finger_dist < 0.5,
                                        torch.abs(object_right_handle_pos[:, 1] - object_left_handle_pos[:, 1]) * 2, up_rew), up_rew)
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - object_right_handle_pos, p=2, dim=-1) * 30, up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # print(right_hand_dist_rew[0])
        # print(left_hand_dist_rew[0])
        # print(up_rew[0])

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset_buf)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["door_open_outward"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        

        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.5,
                        torch.where(left_hand_finger_dist < 0.5,
                                        torch.abs(object_right_handle_pos[:, 1] - object_right_handle_pos[:, 1]) * 2, up_rew), up_rew)
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - object_right_handle_pos, p=2, dim=-1) * 30, up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # print(right_hand_dist_rew[0])
        # print(left_hand_dist_rew[0])
        # print(up_rew[0])

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["door_close_inward"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.5,
                        torch.where(left_hand_finger_dist < 0.5,
                                        1 - torch.abs(object_right_handle_pos[:, 1] - object_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)

        # print(right_hand_dist_rew[0])
        # print(left_hand_dist_rew[0])
        # print(up_rew[0])

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["door_close_outward"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.5,
                        torch.where(left_hand_finger_dist < 0.5,
                                        1 - torch.abs(object_right_handle_pos[:, 1] - object_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 6 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_finger_dist >= 3, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(left_hand_finger_dist >= 3, torch.ones_like(resets), resets)

        # print(right_hand_dist_rew[0])
        # print(left_hand_dist_rew[0])
        # print(up_rew[0])

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["bottle_cap"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_right_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_pos - right_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)

        up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(object_left_handle_pos - object_right_handle_pos, p=2, dim=-1) * 30, up_rew)

        reward = 2.0 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(object_pos[:, 2] <= 0.5, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(right_hand_dist >= 0.5, torch.ones_like(resets), resets)
        resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # print(right_hand_dist_rew[0])
        # print(left_hand_dist_rew[0])
        # print(up_rew[0])

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["block_stack"]:
        # Distance from the hand to the object
        stack_pos1 = target_pos.clone()
        stack_pos2 = target_pos.clone()

        stack_pos1[:, 1] -= 0.1
        stack_pos2[:, 1] -= 0.1
        # stack_pos1[:, 2] += 0.025
        stack_pos1[:, 2] += 0.05

        goal_dist1 = torch.norm(stack_pos1 - object_left_handle_pos, p=2, dim=-1)
        goal_dist2 = torch.norm(stack_pos2 - object_right_handle_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.5,
                        torch.where(left_hand_finger_dist < 0.5,
                            (0.24 - goal_dist1 - goal_dist2) * 2, up_rew), up_rew)

        # up_rew = torch.where(right_hand_finger_dist < 0.75,
        #                 torch.where(left_hand_finger_dist < 0.75,
        #                     torch.where(goal_dist2 < 0.05,
        #                         (0.24 - goal_dist1 - goal_dist2) * 10, up_rew), up_rew), up_rew)

        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)
        stack_rew = torch.zeros_like(right_hand_dist_rew)
        stack_rew = torch.where(goal_dist2 < 0.07,
                        torch.where(goal_dist1 < 0.07,
                            (0.05-torch.abs(stack_pos1[:, 2] - object_left_handle_pos[:, 2])) * 50 ,stack_rew),stack_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 1.5 - right_hand_dist_rew - left_hand_dist_rew + up_rew + stack_rew

        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(right_hand_dist_rew <= 0, torch.ones_like(resets), resets)
        # resets = torch.where(right_hand_finger_dist >= 0.75, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_finger_dist >= 0.75, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["grasp_and_place"]:
        # Distance from the hand to the object
        left_goal_dist = torch.norm(target_pos - object_left_handle_pos, p=2, dim=-1)
        right_goal_dist = torch.norm(target_pos - object_right_handle_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = torch.exp(-10 * right_hand_finger_dist)
        left_hand_dist_rew = torch.exp(-10 * left_hand_finger_dist)

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        # up_rew = torch.zeros_like(right_hand_dist_rew)
        # up_rew = torch.where(right_hand_finger_dist < 0.6,
        #                 torch.where(left_hand_finger_dist < 0.4,
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.exp(-10 * torch.norm(object_right_handle_pos - object_left_handle_pos, p=2, dim=-1)) * 2
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = right_hand_dist_rew + left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_dist_rew < 0, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)

        # # Find out which envs hit the goal and update successes count
        # resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["pen"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = torch.exp(-10 * right_hand_finger_dist)
        left_hand_dist_rew = torch.exp(-10 * left_hand_finger_dist)

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.75,
                        torch.where(left_hand_finger_dist < 0.75,
                            torch.norm(object_right_handle_pos - object_left_handle_pos, p=2, dim=-1) * 5 - 0.8, up_rew), up_rew)
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = up_rew + right_hand_dist_rew + left_hand_dist_rew

        resets = torch.where(torch.abs(up_rew) >= 5, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["push_block"]:
        # Distance from the hand to the object
        left_goal_dist = torch.norm(target_pos - object_left_handle_pos, p=2, dim=-1)
        right_goal_dist = torch.norm(target_pos - object_right_handle_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = (0.8 - (left_goal_dist + right_goal_dist)) * 5
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_dist_rew <= 0, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(right_hand_dist >= 0.5, torch.ones_like(resets), resets)
        resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset_buf)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["re_orientation"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

        goal_another_dist = torch.norm(another_target_pos - another_object_pos, p=2, dim=-1)

        # Orientation alignment for the cube in hand and goal cube
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        quat_another_diff = quat_mul(another_object_rot, quat_conjugate(another_target_rot))
        rot_another_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_another_diff[:, 0:3], p=2, dim=-1), max=1.0))

        dist_rew = goal_dist * dist_reward_scale + goal_another_dist * dist_reward_scale
        rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale + 1.0/(torch.abs(rot_another_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

        # Find out which envs hit the goal and update successes count
        goal_resets = torch.where(torch.abs(rot_dist) < 0.1, torch.ones_like(reset_goal_buf), reset_goal_buf)
        goal_resets = torch.where(torch.abs(rot_another_dist) < 0.1, torch.ones_like(reset_goal_buf), reset_goal_buf)

        successes = successes + goal_resets

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

        # Fall penalty: distance to the goal is larger than a threashold
        reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)
        reward = torch.where(another_object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

        # Check env termination conditions, including maximum success number
        resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(another_object_pos[:, 2] <= 0.2, torch.ones_like(reset_buf), resets)

        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        # Apply penalty for not reaching the goal
        if max_consecutive_successes > 0:
            reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["scissors"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.7,
                        torch.where(left_hand_finger_dist < 0.7,
                            (0.59 - torch.norm(object_right_handle_pos - object_left_handle_pos, p=2, dim=-1)) * 5, up_rew), up_rew)
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 2 + up_rew - right_hand_dist_rew - left_hand_dist_rew

        resets = torch.where(up_rew < -0.5, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(right_hand_finger_dist >= 1.75, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_finger_dist >= 1.75, torch.ones_like(resets), resets)
        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["swing_cup"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale - 1

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(rot_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.4,
                            torch.where(left_hand_finger_dist < 0.4,
                                            rot_rew, up_rew), up_rew)
            
        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(right_hand_dist >= 0.5, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)
        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["switch"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        # up_rew = (right_hand_finger_dist + left_hand_finger_dist)[:, 2] * 20
        up_rew = (1.4-(object_right_handle_pos[:, 2] + object_left_handle_pos[:, 2])) * 5

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_dist_rew <= 0, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(right_hand_dist >= 0.5, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["lift_underarm"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]
        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_right_handle_pos - left_hand_pos, p=2, dim=-1)
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_dist
        left_hand_dist_rew = left_hand_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_dist < 0.08,
                            torch.where(left_hand_dist < 0.08,
                                            3*(0.985 - goal_dist), up_rew), up_rew)
        
        reward = 0.2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(right_hand_dist >= 0.2, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes


def obtain_task_dof_info(task_env, env_ids, num_each_envs, device):
    task_dof_inits = to_torch([0, 0], device=device).repeat((len(env_ids), 1))
    for i in range(len(env_ids)):
        # catch
        if env_ids[i] in range(num_each_envs*0, num_each_envs*5):
            task_dof_inits[i] = to_torch([0, 0], device=device)
        # block
        if env_ids[i] in range(num_each_envs*5, num_each_envs*9):
            task_dof_inits[i] = to_torch([0, 0], device=device)
        # pull door
        if env_ids[i] in range(num_each_envs*12, num_each_envs*14):
            task_dof_inits[i] = to_torch([1.57, 1.57], device=device)
        # switch
        if env_ids[i] in range(num_each_envs*14, num_each_envs*15):
            task_dof_inits[i] = to_torch([0.5, 0.5], device=device)
        
        # scissors
        if env_ids[i] in range(num_each_envs*18, num_each_envs*19):
            task_dof_inits[i] = to_torch([-0.59, -0.59], device=device)
        else:
            task_dof_inits[i] = to_torch([0, 0], device=device)
    
    return task_dof_inits

def obtain_pose_info(task_env, task_righd_body, rigid_body_states, total_rigid_body_tensor, num_each_envs, device):
    object_left_handle_pos, object_right_handle_pos = rigid_body_states[:, 26 * 2 - 1, 0:3].clone(), rigid_body_states[:, 26 * 2 - 1, 0:3].clone()
    object_left_handle_rot, object_right_handle_rot = rigid_body_states[:, 26 * 2 - 1, 3:7].clone(), rigid_body_states[:, 26 * 2 - 1, 3:7].clone()
    
    pointer = 0
    for i, task_name in enumerate(task_env):
        rigid_body_count = task_righd_body[i] * 2 + 52 + 1

        if task_name in ["hand_over", "catch_underarm", "catch_over2underarm", "catch_abreast", "two_catch_underarm", "re_orientation"]:
            l_id_xyz_r_id_xyz = [-1, -0.5, -0.39, -0.04, -1, -0.5, -0.39, -0.04]

        # if task_name in ["block_stack", "push_block"]:
        #     l_id_xyz_r_id_xyz = [0, 0, 0, 0, 1, 0, 0, 0]
        # if task_name in ["grasp_and_place"]:
        #     l_id_xyz_r_id_xyz = [1, 0, 0, 0, 2, 0, 0, 0]

        if task_name in ["block_stack", "push_block"]:
            l_id_xyz_r_id_xyz = [2, 0, 0, 0, 1, 0, 0, 0]
        if task_name in ["grasp_and_place"]:
            l_id_xyz_r_id_xyz = [1, 0, 0, 0, 2, 0, 0, 0]

        if task_name in ["door_open_inward", "door_close_inward"]:
            l_id_xyz_r_id_xyz = [3, -0.5, -0.39, 0.04, 2, -0.5, 0.39, 0.04]
        if task_name in ["door_open_outward", "door_close_outward"]:
            l_id_xyz_r_id_xyz = [3, -0.5, -0.39, -0.04, 2, -0.5, 0.39, -0.04]

        if task_name in ["switch"]:
            l_id_xyz_r_id_xyz = [2, -0.02, -0.05, -0.0, 5, -0.02, -0.05, -0.0]
            # l_id_xyz_r_id_xyz = [2, -0.02, -0.05, -0.0, 5, -0.02, -0.05, -0.0]

        if task_name in ["lift_underarm"]:
            l_id_xyz_r_id_xyz = [0, 0.15, 0.06, -0.0, 0, -0.15, -0.0, 0.06]

        if task_name in ["pen"]:
            l_id_xyz_r_id_xyz = [2, -0.1, 0., 0.0, 1, 0.07, -0.0, 0.0]

        if task_name in ["swing_cup"]:
            l_id_xyz_r_id_xyz = [0, 0, 0.062, 0.0, 1, 0.0, -0.0, 0.06]

        if task_name in ["scissors"]:
            l_id_xyz_r_id_xyz = [2, 0, 0.2, -0.1, 1, 0.0, 0.15, 0.1]

        if task_name in ["bottle_cap"]:
            l_id_xyz_r_id_xyz = [0, 0, 0, 0, 3, 0.0, 0.0, 0.15]
        
        this_task_rigid_body_states = total_rigid_body_tensor[pointer:pointer + num_each_envs*rigid_body_count].reshape(num_each_envs, rigid_body_count, 13)
        object_left_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = this_task_rigid_body_states[:, 26 * 2 + l_id_xyz_r_id_xyz[0], 0:3]
        object_left_handle_rot[num_each_envs*i:num_each_envs*(i+1)] = this_task_rigid_body_states[:, 26 * 2 + l_id_xyz_r_id_xyz[0], 3:7]
        object_left_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = object_left_handle_pos[num_each_envs*i:num_each_envs*(i+1)] + quat_apply(object_left_handle_rot[num_each_envs*i:num_each_envs*(i+1)], to_torch([0, 1, 0], device=device).repeat(num_each_envs, 1) * l_id_xyz_r_id_xyz[1])
        object_left_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = object_left_handle_pos[num_each_envs*i:num_each_envs*(i+1)] + quat_apply(object_left_handle_rot[num_each_envs*i:num_each_envs*(i+1)], to_torch([1, 0, 0], device=device).repeat(num_each_envs, 1) * l_id_xyz_r_id_xyz[2])
        object_left_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = object_left_handle_pos[num_each_envs*i:num_each_envs*(i+1)] + quat_apply(object_left_handle_rot[num_each_envs*i:num_each_envs*(i+1)], to_torch([0, 0, 1], device=device).repeat(num_each_envs, 1) * l_id_xyz_r_id_xyz[3])

        object_right_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = this_task_rigid_body_states[:, 26 * 2 + l_id_xyz_r_id_xyz[4], 0:3]
        object_right_handle_rot[num_each_envs*i:num_each_envs*(i+1)] = this_task_rigid_body_states[:, 26 * 2 + l_id_xyz_r_id_xyz[4], 3:7]
        object_right_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = object_right_handle_pos[num_each_envs*i:num_each_envs*(i+1)] + quat_apply(object_right_handle_rot[num_each_envs*i:num_each_envs*(i+1)], to_torch([0, 1, 0], device=device).repeat(num_each_envs, 1) * l_id_xyz_r_id_xyz[5])
        object_right_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = object_right_handle_pos[num_each_envs*i:num_each_envs*(i+1)] + quat_apply(object_right_handle_rot[num_each_envs*i:num_each_envs*(i+1)], to_torch([1, 0, 0], device=device).repeat(num_each_envs, 1) * l_id_xyz_r_id_xyz[6])
        object_right_handle_pos[num_each_envs*i:num_each_envs*(i+1)] = object_right_handle_pos[num_each_envs*i:num_each_envs*(i+1)] + quat_apply(object_right_handle_rot[num_each_envs*i:num_each_envs*(i+1)], to_torch([0, 0, 1], device=device).repeat(num_each_envs, 1) * l_id_xyz_r_id_xyz[7])
        pointer += num_each_envs*rigid_body_count

    return object_left_handle_pos, object_right_handle_pos, object_left_handle_rot, object_right_handle_rot
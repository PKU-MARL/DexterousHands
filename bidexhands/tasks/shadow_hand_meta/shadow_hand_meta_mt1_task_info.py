from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask

from isaacgym import gymtorch
from isaacgym import gymapi

def obtrain_task_info(task_name):
    if task_name == "catch_underarm_0":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [ 0, -0.79, 0.54, 0, -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_underarm_1":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [0, -0.84, 0.54, 0, -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_underarm_2":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [0.05, -0.79, 0.54, 0., -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_underarm_3":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [-0.05, -0.79, 0.54, -0., -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"
        
    if task_name == "catch_abreast":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
        object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
        goal_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "lift_pot":
        hand_start_pose = [0, 0.05, 0.45, 0, 0, 0]
        another_hand_start_pose = [0, -1.25, 0.45, 0, 0, 3.14159]
        object_pose = [0, -0.6, 0.45, 0, 0, 0]
        goal_pose = [0, -0.39, 1, 0, 0, 0]
        table_pose_dim = [0.0, -0.6, 0.5 * 0.4, 0, 0, 0, 0.3, 0.3, 0.4]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 1000
        object_type = "pot"

    if task_name == "door_open_outward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 0.0, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]                
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
        object_asset_options.vhacd_params.resolution = 200000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "door"

    if task_name == "door_close_inward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 3.14159, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]                
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

    return hand_start_pose, another_hand_start_pose, object_pose, goal_pose, table_pose_dim, object_asset_options, object_type

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, object_left_handle_pos, object_right_handle_pos,
    left_hand_pos, right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, this_task: str
):
    # Distance from the hand to the object
    if this_task in ["catch_underarm", "hand_over", "catch_abreast", "catch_over2underarm"]:
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

        # Orientation alignment for the cube in hand and goal cube
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        dist_rew = goal_dist
        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))

        # Find out which envs hit the goal and update successes count
        goal_resets = torch.where(torch.abs(goal_dist) <= 0.03, torch.ones_like(reset_goal_buf), reset_goal_buf)
        successes = successes + goal_resets

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

        # Fall penalty: distance to the goal is larger than a threashold
        reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

        # Check env termination conditions, including maximum success number
        resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(reset_buf), reset_buf)
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
                                        torch.abs(object_right_handle_pos[:, 1] - object_left_handle_pos[:, 1]) * 2, up_rew), up_rew)
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

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

    if this_task in ["lift_pot"]:
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
        resets = torch.where(right_hand_dist >= 0.2, torch.ones_like(resets), resets)
        resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes
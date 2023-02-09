# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_sarl import process_sarl
from bidexhands.utils.process_marl import process_MultiAgentRL, get_AgentIndex
from bidexhands.utils.process_mtrl import *
from bidexhands.utils.process_metarl import *
from bidexhands.utils.process_offrl import *

def make(task_name, algo):
    set_np_formatting()
    args = get_args(task_name=task_name, algo=algo)
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["mappo", "happo", "hatrpo","maddpg","ippo"]: 
        # maddpg exists a bug now 
        args.task_type = "MultiAgent"

        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    elif args.algo in ["ppo","ddpg","sac","td3","trpo"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    
    elif args.algo in ["mtppo", "random"]:
        args.task_type = "MultiTask"

        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    elif args.algo in ["mamlppo"]:
        args.task_type = "Meta"
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    elif args.algo in ["td3_bc", "bcq", "iql", "ppo_collect"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg, mtppo, random, mamlppo, td3_bc, bcq, iql, ppo_collect]")


    return env

if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))


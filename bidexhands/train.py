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

MARL_ALGOS = ["mappo", "happo", "hatrpo","maddpg","ippo"]
SARL_ALGOS = ["ppo","ddpg","sac","td3","trpo"]
MTRL_ALGOS = ["mtppo", "random"]
META_ALGOS = ["mamlppo"]
OFFRL_ALGOS = ["td3_bc", "bcq", "iql", "ppo_collect"]

def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)
    assert args.algo in MARL_ALGOS + SARL_ALGOS + MTRL_ALGOS + META_ALGOS + OFFRL_ALGOS, \
        "Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo, \
            maddpg,sac,td3,trpo,ppo,ddpg, mtppo, random, mamlppo, td3_bc, bcq, iql, ppo_collect]"
    algo = args.algo
    if args.algo in MARL_ALGOS: 
        # maddpg exists a bug now 
        args.task_type = "MultiAgent"
        algo = "MultiAgentRL"
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        runner = eval('process_{}'.format(algo))(args, env, cfg_train, args.model_dir)
        if args.model_dir != "":
            runner.eval(1000)
        else:
            runner.run()
        return
    elif args.algo in SARL_ALGOS:
        algo = "sarl"
    elif args.algo in MTRL_ALGOS:
        args.task_type = "MultiTask"
    elif args.algo in META_ALGOS:
        args.task_type = "Meta"
    elif args.algo in OFFRL_ALGOS:
        pass 

    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    runner = eval('process_{}'.format(algo))(args, env, cfg_train, logdir)
    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    runner.train(train_epoch=iterations) if args.algo in META_ALGOS else \
        runner.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
        
if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()

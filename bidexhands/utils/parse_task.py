# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from bidexhands.tasks.shadow_hand_over import ShadowHandOver
from bidexhands.tasks.shadow_hand_catch_underarm import ShadowHandCatchUnderarm
from bidexhands.tasks.shadow_hand_two_catch_underarm import ShadowHandTwoCatchUnderarm
from bidexhands.tasks.shadow_hand_catch_abreast import ShadowHandCatchAbreast
from bidexhands.tasks.shadow_hand_lift_underarm import ShadowHandLiftUnderarm
from bidexhands.tasks.shadow_hand_catch_over2underarm import ShadowHandCatchOver2Underarm
from bidexhands.tasks.shadow_hand_door_close_inward import ShadowHandDoorCloseInward
from bidexhands.tasks.shadow_hand_door_close_outward import ShadowHandDoorCloseOutward
from bidexhands.tasks.shadow_hand_door_open_inward import ShadowHandDoorOpenInward
from bidexhands.tasks.shadow_hand_door_open_outward import ShadowHandDoorOpenOutward
from bidexhands.tasks.shadow_hand_bottle_cap import ShadowHandBottleCap
from bidexhands.tasks.shadow_hand_push_block import ShadowHandPushBlock
from bidexhands.tasks.shadow_hand_swing_cup import ShadowHandSwingCup
from bidexhands.tasks.shadow_hand_grasp_and_place import ShadowHandGraspAndPlace
from bidexhands.tasks.shadow_hand_scissors import ShadowHandScissors
from bidexhands.tasks.shadow_hand_switch import ShadowHandSwitch
from bidexhands.tasks.shadow_hand_pen import ShadowHandPen
from bidexhands.tasks.shadow_hand_re_orientation import ShadowHandReOrientation
from bidexhands.tasks.shadow_hand_kettle import ShadowHandKettle
from bidexhands.tasks.shadow_hand_block_stack import ShadowHandBlockStack

# Allegro hand
from bidexhands.tasks.allegro_hand_over import AllegroHandOver
from bidexhands.tasks.allegro_hand_catch_underarm import AllegroHandCatchUnderarm

# Meta
from bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_mt1 import ShadowHandMetaMT1
from bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_ml1 import ShadowHandMetaML1
from bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_mt4 import ShadowHandMetaMT4

from bidexhands.tasks.hand_base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm
from bidexhands.tasks.hand_base.multi_vec_task import MultiVecTaskPython, SingleVecTaskPythonArm
from bidexhands.tasks.hand_base.multi_task_vec_task import MultiTaskVecTaskPython
from bidexhands.tasks.hand_base.meta_vec_task import MetaVecTaskPython
from bidexhands.tasks.hand_base.vec_task_rlgames import RLgamesVecTaskPython

from bidexhands.utils.config import warn_task_name

import json


def parse_task(args, cfg, cfg_train, sim_params, agent_index):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "C++":
        if args.device == "cpu":
            print("C++ CPU")
            task = rlgpu.create_task_cpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskCPU(task, rl_device, False, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))
        else:
            print("C++ GPU")

            task = rlgpu.create_task_gpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskGPU(task, rl_device, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))

    elif args.task_type == "Python":
        print("Python")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        if args.task == "OneFrankaCabinet" :
            env = VecTaskPythonArm(task, rl_device)
        else :
            env = VecTaskPython(task, rl_device)

    elif args.task_type == "MultiAgent":
        print("MultiAgent")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=True)
        except NameError as e:
            print(e)
            warn_task_name()
        env = MultiVecTaskPython(task, rl_device)
    elif args.task_type == "MultiAgent":
        print("Task type: MultiAgent")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=True)
        except NameError as e:
            print(e)
            warn_task_name()
        env = MultiVecTaskPython(task, rl_device)

    elif args.task_type == "MultiTask":
        print("Task type: MultiTask")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env = MultiTaskVecTaskPython(task, rl_device)

    elif args.task_type == "Meta":
        print("Task type: Meta")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env = MetaVecTaskPython(task, rl_device)

    elif args.task_type == "RLgames":
        print("Task type: RLgames")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env = RLgamesVecTaskPython(task, rl_device)
    return task, env



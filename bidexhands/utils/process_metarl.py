# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import torch
from copy import deepcopy

def process_mamlppo(args, env, cfg_train, logdir):
    if args.algo in ["mamlppo"]:
        from bidexhands.algorithms.metarl.maml import Trainer, MAMLPPO, ActorCritic
        learn_cfg = cfg_train["learn"]
        is_testing = learn_cfg["test"]
        # is_testing = True
        # Override resume and testing flags if they are passed as parameters.
        if args.model_dir != "":
            is_testing = True
            chkpt_path = args.model_dir

        logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

        """Set up the PPO system for training or inferencing."""
        actor_critic = ActorCritic(env.observation_space.shape, env.state_space.shape, env.action_space.shape,
                                learn_cfg.get("init_noise_std", 0.8), cfg_train["policy"], asymmetric=(env.num_states > 0))

        pseudo_actor_critic = []
        for i in range(env.task_num):
            pseudo_actor_critic.append(deepcopy(actor_critic))

        inner_algo_ppo = MAMLPPO(vec_env=env,
                pseudo_actor_critic=pseudo_actor_critic,
                num_transitions_per_env=learn_cfg["nsteps"],
                num_learning_epochs=learn_cfg["noptepochs"],
                num_mini_batches=learn_cfg["nminibatches"],
                clip_param=learn_cfg["cliprange"],
                gamma=learn_cfg["gamma"],
                lam=learn_cfg["lam"],
                init_noise_std=learn_cfg.get("init_noise_std", 0.3),
                value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
                entropy_coef=learn_cfg["ent_coef"],
                learning_rate=learn_cfg["optim_stepsize"],
                max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
                use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
                schedule=learn_cfg.get("schedule", "fixed"),
                desired_kl=learn_cfg.get("desired_kl", None),
                model_cfg=cfg_train["policy"],
                device=env.rl_device,
                sampler=learn_cfg.get("sampler", 'sequential'),
                log_dir=logdir,
                is_testing=is_testing,
                print_log=learn_cfg["print_log"],
                apply_reset=False,
                asymmetric=(env.num_states > 0)
                )

        if is_testing and args.model_dir != "":
            print("Loading model from {}".format(chkpt_path))
            inner_algo_ppo.test(chkpt_path)
        elif args.model_dir != "":
            print("Loading model from {}".format(chkpt_path))
            inner_algo_ppo.load(chkpt_path)

        trainer = Trainer(vec_env=env, meta_actor_critic=actor_critic, inner_algo=inner_algo_ppo)
    return trainer

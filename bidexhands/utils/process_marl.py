# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def get_AgentIndex(config):
    agent_index = []
    # right hand
    agent_index.append(eval(config["env"]["handAgentIndex"]))
    # left hand
    agent_index.append(eval(config["env"]["handAgentIndex"]))

    return agent_index
    
def process_MultiAgentRL(args,env, config, model_dir=""):

    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs

    if args.algo in ["mappo", "happo", "hatrpo"]:
        # on policy marl
        from algorithms.marl.runner import Runner
        marl = Runner(vec_env=env,
                    config=config,
                    model_dir=model_dir
                    )
    elif args.algo == 'maddpg':
        # off policy marl
        from algorithms.marl.maddpg.runner import Runner
        marl = Runner(vec_env=env,
            config=config,
            model_dir=model_dir
            )

    return marl

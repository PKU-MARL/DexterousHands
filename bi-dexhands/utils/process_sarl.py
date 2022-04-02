


def process_ppo(args, env, cfg_train, logdir):
    from algorithms.rl.ppo import PPO, ActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

    """Set up the PPO system for training or inferencing."""
    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCritic,
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

    # ppo.test("/home/hp-3070/bi-dexhands/bi-dexhands/logs/shadow_hand_lift_underarm2/ppo/ppo_seed2/model_40000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo.load(chkpt_path)

    return ppo



def process_sac(args, env, cfg_train, logdir):
    from algorithms.rl.sac import SAC, MLPActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    """Set up the SAC system for training or inferencing."""
    sac = SAC(vec_env=env,
              actor_critic=MLPActorCritic,
              ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]]* learn_cfg["hidden_layer"]),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              # clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              entropy_coef = learn_cfg["ent_coef"],
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              reward_scale=learn_cfg["reward_scale"],
              batch_size=learn_cfg["batch_size"],
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
        sac.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        sac.load(chkpt_path)

    return sac


def process_td3(args, env, cfg_train, logdir):
    from algorithms.rl.td3 import TD3, MLPActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    """Set up the TD3 system for training or inferencing."""
    td3 = TD3(vec_env=env,
              actor_critic=MLPActorCritic,
              ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]]* learn_cfg["hidden_layer"]),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              # clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              policy_delay=learn_cfg["policy_delay"],#2,
              act_noise= learn_cfg["act_noise"], #0.1,
              target_noise=learn_cfg["target_noise"], #0.2,
              noise_clip= learn_cfg["noise_clip"], #0.5,
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              reward_scale=learn_cfg["reward_scale"],
              batch_size=learn_cfg["batch_size"],
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
        td3.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        td3.load(chkpt_path)

    return td3


def process_ddpg(args, env, cfg_train, logdir):
    from algorithms.rl.ddpg import DDPG, MLPActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    """Set up the DDPG system for training or inferencing."""
    ddpg = DDPG(vec_env=env,
              actor_critic=MLPActorCritic,
              ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]]* learn_cfg["hidden_layer"]),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              act_noise= learn_cfg["act_noise"], #0.1,
              target_noise=learn_cfg["target_noise"], #0.2,
              noise_clip= learn_cfg["noise_clip"], #0.5,
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              reward_scale=learn_cfg["reward_scale"],
              batch_size=learn_cfg["batch_size"],
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
        ddpg.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ddpg.load(chkpt_path)

    return ddpg


def process_trpo(args, env, cfg_train, logdir):
    from algorithms.rl.trpo import TRPO, ActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = True
        chkpt_path = args.model_dir

    """Set up the TRPO system for training or inferencing."""
    trpo = TRPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              damping =learn_cfg["damping"],
              cg_nsteps =learn_cfg["cg_nsteps"],
              max_kl= learn_cfg["max_kl"],
              max_num_backtrack=learn_cfg["max_num_backtrack"],
              accept_ratio=learn_cfg["accept_ratio"],
              step_fraction=learn_cfg["step_fraction"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
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
        trpo.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        trpo.load(chkpt_path)

    return trpo
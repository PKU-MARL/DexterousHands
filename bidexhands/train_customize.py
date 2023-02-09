import bidexhands as bi
import torch

env_name = 'ShadowHandOver'
algo = "ppo"
env = bi.make(env_name, algo)

obs = env.reset()
terminated = False

while not terminated:
    act = torch.tensor(env.action_space.sample()).repeat((env.num_envs, 1))
    obs, reward, done, info = env.step(act)

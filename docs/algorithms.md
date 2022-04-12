## Algorithms

We give a brief introduction to the single/multi-agent reforcement learning algorithm here. For more details, please refer to the original paper.

### Single-agent reinforcement learning algorithms

#### Trust Region Policy Optimization

TRPO is a basic policy optimization algorithm, with theoretically justified monotonic improvement. Based on the theorem1 in the original paper by John Schulman et. al. To yield a practical algorithm,TRPO made several approximations including 
solving the optimization problem with conjugate gradient algorithm followed by a line search


#### Proximal Policy Optimization
PPO is a policy optimization algorithm enjoying simpler implementation, more general application and better sample complexity over TRPO.

#### Deep Deterministic Policy Gradient
DDPG, based on the DPG algorithm, is a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces.

#### Twin Delayed Deep Deterministic policy gradient
TD3 is an actor-critic algorithm which applies its modifications to the state of the art actor-critic method for continuous control, DDPG. It focused on two outcomes that occur as the result of estimation error, overestimation bias and a high variance build-up. 

### Multi-agent reinforcement learning algorithms

#### Heterogeneous-Agent Proximal Policy Optimisation
HAPPO is a multi-agent policy optimization algorithm that follows the centralized training decentralized execution (CTDE) paradigm. HAPPO doesn't assume homogeneous agents and doesn't require decomposibility of the joint value function. The theoretical core of extending PPO to multi-agent settings is the advantage decomposition lemma(Lemma 1 in the original paper). 

#### Multi-Agent Proximal Policy Optimization
MAPPO (Multi-Agent PPO) is an application of the actor-critic single-agent PPO algorithm to multi-agent tasks. It follows the CTDE structure.
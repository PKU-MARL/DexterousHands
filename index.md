# Bi-DexHands: Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning

## Introduction

Achieving human-level dexterity is an important open problem in robotics. However, tasks of dexterous hand manipulation, even at the baby level, are challenging to solve through reinforcement learning (RL). 

The difficulty lies in the high degrees of freedom and the required cooperation among heterogeneous agents (e.g., joints of fingers).  
In this study, we propose the **Bi**manual **Dex**terous **Hands** Benchmark (Bi-DexHands), a simulator that involves two dexterous hands with tens of bimanual manipulation tasks and thousands of target objects. Specifically, tasks in Bi-DexHands are designed to match different levels of human motor skills according to cognitive science literature. We built Bi-DexHands  in the Issac Gym; this enables highly efficient RL training,  reaching 30,000+ FPS by only one single NVIDIA RTX 3090. 

We provide a comprehensive benchmark for popular RL algorithms under different settings; this includes Single-agent/Multi-agent RL, Offline RL, Multi-task RL, and Meta RL. Our results show that the PPO type of on-policy algorithms can master simple manipulation tasks that are equivalent up to 48-month human babies (e.g., catching a flying object, opening a bottle), while multi-agent RL can further help to master manipulations that require skilled bimanual cooperation (e.g., lifting a pot, stacking blocks).  
Despite the success on each single task, when it comes to acquiring multiple manipulation skills, existing RL algorithms fail to work in most of the multi-task and the few-shot learning settings, which calls for more substantial development from the RL community. 

Our project is open sourced at [https://github.com/PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands).

## Framework

![Framework](./assets/images/overview.png)

## Results

![Results](./assets/images/merge_20.png)

## Demo

![Demo](./assets/images/quick_demo_v2.gif)

## Code

Please see [our github repo](https://github.com/PKU-MARL/DexterousHands) for code and data of this project.

## Citation


## Contact

Bi-DexHands is a project contributed by [Yuanpei Chen](https://github.com/cypypccpy), [Shengjie Wang](https://github.com/Shengjie-bob), [Hao Dong](https://zsdonghao.github.io), [Zongqing Lu](https://z0ngqing.github.io), [Yaodong Yang](https://www.yangyaodong.com/) at Peking University, please contact yaodong.yang@pku.edu.cn if you are interested to collaborate.
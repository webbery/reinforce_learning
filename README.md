### 强化学习示例介绍

示例如有不能运行的问题，请反馈

| 强化学习算法  | 使用游戏 | 神经网络 | 优化 | 动作空间 |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| [QTable](rainbow/00_FrozenLake_QTable.ipynb)      | FrozenLake-v1 | 无 | 无 | 无 | 
| [DQN](rainbow/01_FrozenLake_DQN.ipynb)         | FrozenLake-v1  | 两层Linear+一层Output | 无 | 离散动作空间 | 
| [SARSA](rainbow/02_FrozenLake_SARSA.ipynb)       | FrozenLake-v1  | 两层Linear+一层Output | 无 | 离散动作空间 | 
| [SARSA](rainbow/03_CartPole_ReplayBuffer.ipynb)        | CartPole-v1  | 两层Linear+一层Output | ReplayBuffer | 离散动作空间 |
| [Reinforce](policy/01-Reinforce_MountainCar.ipynb)   | CartPole-v1  | 两层Linear+一层Output | baseline |  离散动作空间  |
| [ActorCritic](policy/02-ActorCritic_CartPole.ipynb) | CartPole-v1  | 两层Linear+一层Output | 无 |  离散动作空间  |
| [Reinforce](policy/03-BaselineReinforce-Pendulum.ipynb) | Pendulum-v1  | 两层Linear+一层Output | baseline | 连续动作空间 |
| [A2C](policy/04-A2C-Pendulum.ipynb) | Pendulum-v1  | 两层Linear+一层Output | 无 | 连续动作空间 |
| [DPG](policy/05-DPG-Pendulum.ipynb) | Pendulum-v1  | 两层Linear+一层Output | ReplayBuffer | 连续动作空间 |
| [Reinforce](policy/05-reinforce-Pendulum.ipynb) | Pendulum-v1  | 两层Linear+一层Output | ReplayBuffer | 连续动作空间 |
| [TD3](policy/06-TD3-Pendulum.ipynb) | Pendulum-v1  | 两层Linear+一层Output | ReplayBuffer/目标网络/截断双Q学习/目标策略网络中加入噪声 | 连续动作空间 |

最后提供了一个强化学习仿真器示例[BreakEnv](BreakEnv.py)，支持录制视频。该示例主要用于高精地图车道组打断，但奖励算法没有全部完成，仅供参考学习

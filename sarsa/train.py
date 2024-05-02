#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import gym
from gridworld import CliffWalkingWapper, FrozenLakeWapper, GridWorld
from agent import SarsaAgent
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')

# assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"


def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）

    # edit!!!
    obs = obs[0]

    action = agent.sample(obs)  # 根据算法选择一个动作

    while True:
        # print(env.step(action))
        # next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        next_obs, reward, done, _, _ = env.step(action)
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作
        # 训练 Sarsa 算法
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    obs = obs[0]
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def smooth(data, weight=0.9):
    '''用于平滑曲线
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_curves(steps, rewards):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.subplot(2, 1, 1)
    plt.title("steps training curve of q-learning")
    # plt.xlim(0, len(steps), 10)  # 设置x轴的范围
    plt.xlim(left=0, right=len(steps), emit=True)  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.ylabel('steps')
    plt.plot(steps, label='steps')
    plt.plot(smooth(steps), label='smoothed')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("rewards training curve of q-learning")
    # plt.xlim(0, len(steps), 10)  # 设置x轴的范围
    plt.xlim(left=0, right=len(rewards), emit=True)  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.ylabel('rewards')
    plt.plot(rewards, label='steps')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()

    plt.show()


def main():
    # env = gym.make("FrozenLake-v1", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)

    gridmap = [
        'SFFHF',
        'FFFHF',
        'HHFFG',
        'FFFFF',
        'FFHHH']
    env = GridWorld(gridmap)

    # env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    # env = CliffWalkingWapper(env)

    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    # edit！！！
    rewards = []
    steps = []

    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))
        # edit!!!
        rewards.append(ep_reward)
        steps.append(ep_steps)

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    # 训练结束，查看算法效果
    test_episode(env, agent)
    # plot_curves(rewards, title=f"rewards training curve of q-learning")
    plot_curves(steps, rewards)


if __name__ == "__main__":
    main()

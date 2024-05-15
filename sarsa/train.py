import gym
from gridworld import GridWorld
from agent import SarsaAgent
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

matplotlib.use('TkAgg')


def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    obs = env.reset()

    obs = obs[0]

    action = agent.sample(obs)

    while True:
        next_obs, reward, done, _, _ = env.step(action)
        next_action = agent.sample(next_obs)
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
    plt.title("training curve of SARSA")
    plt.xlim(left=0, right=len(steps), emit=True)  # 设置x轴的范围
    plt.xticks(np.arange(0, len(steps), 20))
    plt.xlabel('epsiodes')
    plt.ylabel('steps/rewards')
    plt.plot(steps, label='steps', color='red')
    plt.plot(smooth(steps), label='smoothed-steps', color='green')
    plt.plot(rewards, label='rewards', color='blue')
    plt.plot(smooth(rewards), label='smoothed-rewards', color='black')
    plt.legend()

    plt.show()


def main():
    gridmap = [
        'FFFHF',
        'FFFHF',
        'HHFFG',
        'FFFFF',
        'FFHHH']
    env = GridWorld(gridmap)

    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.9)

    rewards = []
    steps = []

    is_render = False
    for episode in range(400):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))
        rewards.append(ep_reward)
        steps.append(ep_steps)

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    # 训练结束，查看算法效果
    test_episode(env, agent)
    plot_curves(steps, rewards)


if __name__ == "__main__":
    main()

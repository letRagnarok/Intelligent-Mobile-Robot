import gym
import numpy as np
import time
from gym import wrappers
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
import pandas as pd
from collections import defaultdict
import datetime
import argparse
import seaborn as sns
import math

matplotlib.use('TkAgg')
PIXEL = 80  # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


# 迷宫GUI界面
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_action = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * PIXEL, MAZE_H * PIXEL))
        self._build_maze()

    def _build_maze(self):  # 画迷宫最开始的样子
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * PIXEL,
                                width=MAZE_W * PIXEL)
        for c in range(PIXEL, MAZE_W * PIXEL, PIXEL):  # draw column lines
            x0, y0, x1, y1 = c, 0, c, MAZE_H * PIXEL
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(PIXEL, MAZE_H * PIXEL, PIXEL):  # draw row lines
            x0, y0, x1, y1 = 0, r, MAZE_W * PIXEL, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([40, 40])  # 中心坐标

        # 画初始点
        self.agent = self.canvas.create_oval(
            origin[0] - 35, origin[1] - 35,
            origin[0] + 35, origin[1] + 35,
            fill='brown')

        # 画死亡区域
        hell1_center = origin + np.array([0, PIXEL * 2])
        self.hell1 = self.canvas.create_oval(
            hell1_center[0] - 35, hell1_center[1] - 35,
            hell1_center[0] + 35, hell1_center[1] + 35,
            fill='black')

        hell2_center = origin + np.array([PIXEL, PIXEL * 2])
        self.hell2 = self.canvas.create_oval(
            hell2_center[0] - 35, hell2_center[1] - 35,
            hell2_center[0] + 35, hell2_center[1] + 35,
            fill='black')

        hell3_center = origin + np.array([PIXEL * 3, 0])
        self.hell3 = self.canvas.create_oval(
            hell3_center[0] - 35, hell3_center[1] - 35,
            hell3_center[0] + 35, hell3_center[1] + 35,
            fill='black')

        hell4_center = origin + np.array([PIXEL * 3, PIXEL])
        self.hell4 = self.canvas.create_oval(
            hell4_center[0] - 35, hell4_center[1] - 35,
            hell4_center[0] + 35, hell4_center[1] + 35,
            fill='black')

        hell5_center = origin + np.array([PIXEL * 2, PIXEL * 4])
        self.hell5 = self.canvas.create_oval(
            hell5_center[0] - 35, hell5_center[1] - 35,
            hell5_center[0] + 35, hell5_center[1] + 35,
            fill='black')

        hell6_center = origin + np.array([PIXEL * 3, PIXEL * 4])
        self.hell6 = self.canvas.create_oval(
            hell6_center[0] - 35, hell6_center[1] - 35,
            hell6_center[0] + 35, hell6_center[1] + 35,
            fill='black')

        hell7_center = origin + np.array([PIXEL * 4, PIXEL * 4])
        self.hell7 = self.canvas.create_oval(
            hell7_center[0] - 35, hell7_center[1] - 35,
            hell7_center[0] + 35, hell7_center[1] + 35,
            fill='black')

        # 画终点
        endpoint_center = origin + np.array([PIXEL * 4, PIXEL * 2])
        self.endpoint = self.canvas.create_oval(
            endpoint_center[0] - 35, endpoint_center[1] - 35,
            endpoint_center[0] + 35, endpoint_center[1] + 35,
            fill='yellow')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        origin = np.array([40, 40])
        self.agent = self.canvas.create_oval(
            origin[0] - 35, origin[1] - 35,
            origin[0] + 35, origin[1] + 35,
            fill='brown')
        # return observation
        return self.canvas.coords(self.agent)

    def step(self, action):
        s = self.canvas.coords(self.agent)  # 获取当前agent的坐标
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > PIXEL:
                base_action[1] -= PIXEL
        elif action == 1:  # down
            if s[1] + PIXEL < MAZE_H * PIXEL:
                base_action[1] += PIXEL
        elif action == 2:  # left
            if s[0] > PIXEL:
                base_action[0] -= PIXEL
        elif action == 3:  # right
            if s[0] + PIXEL < MAZE_W * PIXEL:
                base_action[0] += PIXEL

        self.canvas.move(self.agent, base_action[0], base_action[1])

        next_state = self.canvas.coords(self.agent)

        if next_state == self.canvas.coords(self.endpoint):  # 到达终点
            reward = 1
            done = True
            # next_state = 'terminal'
            next_state = 'terminal'
        elif next_state in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2),
                            self.canvas.coords(self.hell3),
                            self.canvas.coords(self.hell4), self.canvas.coords(self.hell5),
                            self.canvas.coords(self.hell6),
                            self.canvas.coords(self.hell7)]:
            reward = -1
            done = True
            # next_state = 'terminal'
            next_state = 'terminal'
        else:
            reward = 0
            done = False

        return next_state, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


class sarsa:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, next_state, next_action):
        self.check_state_exist(next_state)
        predict_q = self.q_table.loc[s, a]
        if next_state != 'terminal':
            target_q = r + self.gamma * self.q_table.loc[next_state, next_action]
        else:
            target_q = r
        self.q_table.loc[s, a] += self.lr * (target_q - predict_q)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


def update():
    print("开始训练！")
    for episode in range(episodes):
        observation = env.reset()

        ep_step = 0

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            next_observation, reward, done = env.step(action)
            next_action = RL.choose_action(str(next_observation))
            RL.learn(str(observation), action, reward, str(next_observation), next_action)
            observation = next_observation
            ep_step += 1

            if done:
                break
        steps.append(ep_step)
        if (episode + 1) % 20 == 0:
            print(f"回合: {episode + 1}/{episodes},步数：{ep_step:.1f}")
    print("完成训练！")


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


def plot_steps(steps, title="learning curve"):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    # plt.xlim(0, len(steps), 10)  # 设置x轴的范围
    plt.xlim(left=0, right=len(steps), emit=True)  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.plot(steps, label='steps')
    plt.plot(smooth(steps), label='smoothed')
    plt.legend()
    plt.show()


episodes = 140
steps = []
env = Maze()
RL = sarsa(actions=list(range(env.n_action)))
env.after(100, update)
env.mainloop()
plot_steps(steps, title=f"training curve of SARSA")

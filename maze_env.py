"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import time
import sys

if sys.version_info.major == 2:  # python 2.0
    import Tkinter as tk
else:  # python 3.0
    import tkinter as tk

UNIT = 80  # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 上下左右
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(UNIT, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([40, 40])

        # hell
        hell1_center = origin + np.array([0, UNIT * 2])
        self.hell1 = self.canvas.create_oval(
            hell1_center[0] - 35, hell1_center[1] - 35,
            hell1_center[0] + 35, hell1_center[1] + 35,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_oval(
            hell2_center[0] - 35, hell2_center[1] - 35,
            hell2_center[0] + 35, hell2_center[1] + 35,
            fill='black')
        # hell
        hell3_center = origin + np.array([UNIT * 3, 0])
        self.hell3 = self.canvas.create_oval(
            hell3_center[0] - 35, hell3_center[1] - 35,
            hell3_center[0] + 35, hell3_center[1] + 35,
            fill='black')
        # hell
        hell4_center = origin + np.array([UNIT * 3, UNIT])
        self.hell4 = self.canvas.create_oval(
            hell4_center[0] - 35, hell4_center[1] - 35,
            hell4_center[0] + 35, hell4_center[1] + 35,
            fill='black')
        # hell
        hell5_center = origin + np.array([UNIT * 2, UNIT * 4])
        self.hell5 = self.canvas.create_oval(
            hell5_center[0] - 35, hell5_center[1] - 35,
            hell5_center[0] + 35, hell5_center[1] + 35,
            fill='black')
        # hell
        hell6_center = origin + np.array([UNIT * 3, UNIT * 4])
        self.hell6 = self.canvas.create_oval(
            hell6_center[0] - 35, hell6_center[1] - 35,
            hell6_center[0] + 35, hell6_center[1] + 35,
            fill='black')
        # hell
        hell7_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.hell7 = self.canvas.create_oval(
            hell7_center[0] - 35, hell7_center[1] - 35,
            hell7_center[0] + 35, hell7_center[1] + 35,
            fill='black')

        # create oval
        oval_center = origin + np.array([UNIT * 4, UNIT * 2])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 35, oval_center[1] - 35,
            oval_center[0] + 35, oval_center[1] + 35,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_oval(
            origin[0] - 35, origin[1] - 35,
            origin[0] + 35, origin[1] + 35,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([40, 40])
        self.rect = self.canvas.create_oval(
            origin[0] - 35, origin[1] - 35,
            origin[0] + 35, origin[1] + 35,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)  # 获取当前agent的坐标
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6),
                    self.canvas.coords(self.hell7)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()

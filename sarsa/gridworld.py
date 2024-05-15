# -*- coding: utf-8 -*-
import random

import gym
import turtle
import numpy as np


def randomize_S(gridmap):
    rows, cols = len(gridmap), len(gridmap[0])
    available_positions = [(i, j) for i in range(rows) for j in range(cols)
                           if gridmap[i][j] != 'G' and gridmap[i][j] != 'H']
    random_position = random.choice(available_positions)
    new_row = list(gridmap[random_position[0]])
    new_row[random_position[1]] = 'S'  # 在选定的位置放置'S'
    gridmap[random_position[0]] = ''.join(new_row)  # 将列表转换回字符串
    print(gridmap)


def GridWorld(gridmap=None, is_slippery=False):
    if gridmap is None:
        gridmap = ['SFFHF', 'FFFHF', 'HHFFG', 'FFFFF', 'FFHHH']
        # gridmap = ['FFFHF', 'FFFHF', 'HHFFG', 'FFFFF', 'FFHHH']
    randomize_S(gridmap)
    env = gym.make("FrozenLake-v1", desc=gridmap, is_slippery=False)
    env = FrozenLakeWapper(env)
    return env


class FrozenLakeWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.max_y = env.desc.shape[0]
        self.max_x = env.desc.shape[1]
        self.t = None
        self.unit = 50

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for _ in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for i in range(self.desc.shape[0]):
                for j in range(self.desc.shape[1]):
                    x = j
                    y = self.max_y - 1 - i
                    if self.desc[i][j] == b'S':  # Start
                        self.draw_box(x, y, 'white')
                    elif self.desc[i][j] == b'F':  # Frozen ice
                        self.draw_box(x, y, 'white')
                    elif self.desc[i][j] == b'G':  # Goal
                        self.draw_box(x, y, 'yellow')
                    elif self.desc[i][j] == b'H':  # Hole
                        self.draw_box(x, y, 'black')
                    else:
                        self.draw_box(x, y, 'white')
            # self.t.shape('turtle')
            self.t.shape('circle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)


if __name__ == '__main__':
    # 自定义格子世界，S为出发点Start, F为平地Floor, H为洞Hole, G为出口目标Goal
    gridmap = [
        'SFFHF',
        'FFFHF',
        'HHFFG',
        'FFFFF',
        'FFHHH']
    env = GridWorld(gridmap)

    env.reset()
    for step in range(10):
        action = np.random.randint(0, 4)
        obs, reward, done, info, _ = env.step(action)
        print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format( \
            step, action, obs, reward, done, info))

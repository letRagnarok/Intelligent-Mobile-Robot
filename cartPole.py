import gym
import time

env = gym.make('CartPole-v1', render_mode="human")

observation = env.reset()
count = 0
for t in range(100):
    action = env.action_space.sample()
    observation, reward, done, info, _ = env.step(action)
    if done:
        break
    env.render()
    count += 1
    time.sleep(0.2)
print(count)

env.close()

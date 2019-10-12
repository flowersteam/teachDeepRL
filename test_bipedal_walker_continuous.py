import gym
import gym_flowers
import time
import numpy as np

env = gym.make('bipedal-walker-continuous-v0')
walker_types = ['short', 'default', 'quadru']
for w_type in walker_types:
    env.env.my_init({'leg_size': w_type})  # set walker type
    env.set_environment(stump_height=np.random.uniform(0,3), obstacle_spacing=np.random.uniform(0,6)) # Stump Tracks
    #env.set_environment(poly_shape=np.random.uniform(0,4,12))  # Hexagon Tracks
    env.reset()
    for i in range(100):
        env.step(env.env.action_space.sample())
        env.render()
        time.sleep(0.01)
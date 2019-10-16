import gym
import teachDRL.gym_flowers
import time
import numpy as np

env = gym.make('bipedal-walker-continuous-v0')
walker_types = ['short', 'default', 'quadru', 'quadru']
for i,w_type in enumerate(walker_types):
    env.env.my_init({'leg_size': w_type})  # set walker type
    env.set_environment(stump_height=np.random.uniform(0,3), obstacle_spacing=np.random.uniform(0,6)) # Stump Tracks
    if i == len(walker_types)-1:
        env.set_environment(poly_shape=np.random.uniform(0,4,12))  # Hexagon Tracks
    env.reset()
    for i in range(250):
        env.step(env.env.action_space.sample())
        env.render()
        time.sleep(0.01)
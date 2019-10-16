from gym.envs.registration import register

register(
    id='bipedal-walker-continuous-v0',
    entry_point='teachDRL.gym_flowers.envs:BipedalWalkerContinuous',
    max_episode_steps=2000,
    reward_threshold=300,
)
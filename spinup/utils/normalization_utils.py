import numpy as np

class MaxMinFilter():
    def __init__(self, env_params_dict=None):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 1.0
        self.new_mind = -1.0

        #parameterized env specific
        self.env_p_dict = env_params_dict
        if env_params_dict:
            assert(env_params_dict['roughness'] is None
                   and env_params_dict['gap_width'] is None
                   and env_params_dict['step_height'] is None
                   and env_params_dict['step_number'] is None
                   and env_params_dict['stump_height'] is not None)
            self.min_stump = env_params_dict['stump_height'][0]
            self.max_stump = env_params_dict['stump_height'][1]

    def __call__(self, x):
        new_env_params = []
        if self.env_p_dict:
            env_params = x[0:2].clip(self.min_stump, self.max_stump)
            new_env_params = (((env_params - self.min_stump) * (self.new_maxd - self.new_mind)
                        ) / (self.max_stump - self.min_stump)) + self.new_mind
            x = x[2:]
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return np.concatenate((new_env_params, new_obs))
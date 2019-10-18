import numpy as np
import pickle
from teachDRL.teachers.utils.test_utils import get_test_set_name
import copy

np.random.seed(4222)
sample_size = 50
arg_ranges = {'roughness':None,
              'stump_height':None,
              'stump_width':None,
              'obstacle_spacing':None,
              'poly_shape':None,
              'stump_seq':None}

name = get_test_set_name(arg_ranges)
test_env_list = []
# --- sample environments uniformly
points = []
for i in range(sample_size):
    test_env = arg_ranges.copy()
    point = []
    for k,v in test_env.items():
        if v is not None:
            if k == "poly_shape":
                point.append(np.random.uniform(v[0],v[1],12))
            elif k == 'stump_seq':
                point.append(np.random.uniform(v[0], v[1], 10))
            else:
                point.append(np.random.uniform(v[0], v[1]))
            test_env[k] = point[-1]
    points.append(point)
    test_env_list.append(test_env)

# if len(points[0]) <= 2:
#     import matplotlib.pyplot as plt
#     plt.plot([p[0] for p in points], [p[1] for p in points], 'o')
#     plt.axis("equal")
#     plt.show()

print(test_env_list)
pickle.dump(test_env_list, open('test_sets/'+name+".pkl", "wb"))
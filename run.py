import argparse
from teachDRL.spinup.utils.run_utils import setup_logger_kwargs
from teachDRL.spinup.algos.sac.sac import sac
from teachDRL.spinup.algos.sac import core
import gym
import teachDRL.gym_flowers
from teachDRL.teachers.teacher_controller import TeacherController
from collections import OrderedDict
import os
import numpy as np

# Argument definition
parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--seed', '-s', type=int, default=0)

# Deep RL student arguments, so far only works with SAC
parser.add_argument('--hid', type=int, default=-1)  # number of neurons in hidden layers
parser.add_argument('--l', type=int, default=1)  # number of hidden layers
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--gpu_id', type=int, default=-1)  # default is no GPU
parser.add_argument('--ent_coef', type=float, default=0.005)
parser.add_argument('--max_ep_len', type=int, default=2000)
parser.add_argument('--steps_per_ep', type=int, default=200000)
parser.add_argument('--buf_size', type=int, default=2000000)
parser.add_argument('--nb_test_episodes', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--train_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1000)

# Parameterized bipedal walker arguments, so far only works with bipedal-walker-continuous-v0
parser.add_argument('--env', type=str, default="bipedal-walker-continuous-v0")

# Choose student (walker morphology)
parser.add_argument('--leg_size', type=str, default="default")  # choose walker type ("short", "default" or "quadru")


# Selection of parameter space
# So far 3 choices: "--max_stump_h 3.0 --max_obstacle_spacing 6.0" (aka Stump Tracks) or "-hexa" (aka Hexagon Tracks)
# or "-seq" (untested experimental env)
parser.add_argument('--max_stump_h', type=float, default=None)
parser.add_argument('--max_stump_w', type=float, default=None)
parser.add_argument('--max_stump_r', type=float, default=None)
parser.add_argument('--roughness', type=float, default=None)
parser.add_argument('--max_obstacle_spacing', type=float, default=None)
parser.add_argument('--max_gap_w', type=float, default=None)
parser.add_argument('--step_h', type=float, default=None)
parser.add_argument('--step_nb', type=float, default=None)
parser.add_argument('--hexa_shape', '-hexa', action='store_true')
parser.add_argument('--stump_seq', '-seq', action='store_true')

# Teacher-specific arguments:
parser.add_argument('--teacher', type=str, default="ALP-GMM")  # ALP-GMM, Covar-GMM, RIAC, Oracle, Random

# ALPGMM (Absolute Learning Progress - Gaussian Mixture Model) related arguments
parser.add_argument('--gmm_fitness_fun', '-fit', type=str, default=None)
parser.add_argument('--nb_em_init', type=int, default=None)
parser.add_argument('--min_k', type=int, default=None)
parser.add_argument('--max_k', type=int, default=None)
parser.add_argument('--fit_rate', type=int, default=None)
parser.add_argument('--weighted_gmm', '-wgmm', action='store_true')
parser.add_argument('--alp_max_size', type=int, default=None)

# CovarGMM related arguments
parser.add_argument('--absolute_lp', '-alp', action='store_true')

# RIAC related arguments
parser.add_argument('--max_region_size', type=int, default=None)
parser.add_argument('--alp_window_size', type=int, default=None)

args = parser.parse_args()

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

# Bind this run to specific GPU if there is one
if args.gpu_id != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Set up Student's DeepNN architecture if provided
ac_kwargs = dict()
if args.hid != -1:
    ac_kwargs['hidden_sizes'] = [args.hid] * args.l

# Set bounds for environment's parameter space format:[min, max, nb_dimensions] (if no nb_dimensions, assumes only 1)
param_env_bounds = OrderedDict()
if args.max_stump_h is not None:
    param_env_bounds['stump_height'] = [0, args.max_stump_h]
if args.max_stump_w is not None:
    param_env_bounds['stump_width'] = [0, args.max_stump_w]
if args.max_stump_r is not None:
    param_env_bounds['stump_rot'] = [0, args.max_stump_r]
if args.max_obstacle_spacing is not None:
    param_env_bounds['obstacle_spacing'] = [0, args.max_obstacle_spacing]
if args.hexa_shape:
    param_env_bounds['poly_shape'] = [0, 4.0, 12]
if args.stump_seq:
    param_env_bounds['stump_seq'] = [0, 6.0, 10]

# Set Teacher hyperparameters
params = {}
if args.teacher == 'ALP-GMM':
    if args.gmm_fitness_fun is not None:
        params['gmm_fitness_fun'] = args.gmm_fitness_fun
    if args.min_k is not None and args.max_k is not None:
        params['potential_ks'] = np.arange(args.min_k, args.max_k, 1)
    if args.weighted_gmm is True:
        params['weighted_gmm'] = args.weighted_gmm
    if args.nb_em_init is not None:
        params['nb_em_init'] = args.nb_em_init
    if args.fit_rate is not None:
        params['fit_rate'] = args.fit_rate
    if args.alp_max_size is not None:
        params['alp_max_size'] = args.alp_max_size
elif args.teacher == 'Covar-GMM':
    if args.absolute_lp is True:
        params['absolute_lp'] = args.absolute_lp
elif args.teacher == "RIAC":
    if args.max_region_size is not None:
        params['max_region_size'] = args.max_region_size
    if args.alp_window_size is not None:
        params['alp_window_size'] = args.alp_window_size
elif args.teacher == "Oracle":
    if 'stump_height' in param_env_bounds and 'obstacle_spacing' in param_env_bounds:
        params['window_step_vector'] = [0.1, -0.2]  # order must match param_env_bounds construction
    elif 'poly_shape' in param_env_bounds:
        params['window_step_vector'] = [0.1] * 12
        print('hih')
    elif 'stump_seq' in param_env_bounds:
        params['window_step_vector'] = [0.1] * 10
    else:
        print('Oracle not defined for this parameter space')
        exit(1)

env_f = lambda: gym.make(args.env)
env_init = {}
env_init['leg_size'] = args.leg_size


# Initialize teacher
Teacher = TeacherController(args.teacher, args.nb_test_episodes, param_env_bounds,
                            seed=args.seed, teacher_params=params)

# Launch Student training
sac(env_f, actor_critic=core.mlp_actor_critic, ac_kwargs=ac_kwargs, gamma=args.gamma, seed=args.seed, epochs=args.epochs,
    logger_kwargs=logger_kwargs, alpha=args.ent_coef, max_ep_len=args.max_ep_len, steps_per_epoch=args.steps_per_ep,
    replay_size=args.buf_size, env_init=env_init, env_name=args.env, nb_test_episodes=args.nb_test_episodes, lr=args.lr,
    train_freq=args.train_freq, batch_size=args.batch_size, Teacher=Teacher)
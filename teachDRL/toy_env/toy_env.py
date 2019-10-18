import numpy as np
import scipy.stats as sp
import time
from teachDRL.teachers.algos.riac import RIAC
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from teachDRL.teachers.algos.covar_gmm import CovarGMM
from teachDRL.teachers.utils.plot_utils import region_plot_gif, gmm_plot_gif, random_plot_gif
import matplotlib.pyplot as plt
import pickle
import copy
import sys
from collections import OrderedDict
import seaborn as sns; sns.set()

# A simple n-dimensional toy parameter space to test teacher algorithms
class ToyEnv(object):  # n-dimensional grid
    def __init__(self, nb_cubes=10, nb_dims=2, noise=0.0):
        self.nb_cubes = nb_cubes  # Number of hypercubes per dimensions
        self.nb_dims = nb_dims  # Number of dimensions

        self.nb_total_cubes = nb_cubes ** nb_dims
        self.step_size = 1/nb_cubes
        self.bnds = [np.arange(0,1+self.step_size,self.step_size) for i in range(nb_dims)]
        self.params = []
        self.cube_competence = np.zeros((nb_cubes, ) * nb_dims)
        self.noise = noise
        self.max_per_cube = 100

    def reset(self):
        self.cube_competence = np.zeros((nb_cubes,) * nb_dims)
        self.params = []

    def get_score(self):  # Returns the percentage of "mastered" hypercubes (A cube is "mastered" if its competence >75)
        score = np.where(self.cube_competence > (3*(self.max_per_cube/4)))  #
        return (len(score[0]) / self.nb_total_cubes)*100

    def episode(self, param):
        # Ensure param values fall in bounds
        for v in param:
            if (v < 0.0) or (v > 1.0):
                print('param is out of bounds')
                exit(1)
        p = param[0:self.nb_dims]  # discard potential useless dimensions
        self.params.append(p)

        # 1 - Find in which hypercube the parameter vector falls
        arr_p = np.array([p])
        cubes = sp.binned_statistic_dd(arr_p, np.ones(arr_p.shape), 'count',
                                       bins=self.bnds).statistic
        cube_idx = tuple([v[0] for v in cubes[0].nonzero()])

        # 2 - Check if hypercube is "unlocked" by checking if a previous adjacent neighbor is unlocked
        if all(v == 0 for v in cube_idx):  # If initial cube, no need to have unlocked neighbors to learn
            self.cube_competence[cube_idx] = min(self.cube_competence[cube_idx] + 1, self.max_per_cube)
        else: # Find index of previous adjacent neighboring hypercubes
            prev_cube_idx = [[idx, max(0, idx - 1)] for idx in cube_idx]
            previous_neighbors_idx = np.array(np.meshgrid(*prev_cube_idx)).T.reshape(-1,len(prev_cube_idx))
            for pn_idx in previous_neighbors_idx:
                prev_idx = tuple(pn_idx)
                if all(v == cube_idx[i] for i,v in enumerate(prev_idx)):  # Original hypercube, not previous neighbor
                    continue
                else:
                    if self.cube_competence[prev_idx] >= (3*(self.max_per_cube/4)):  # Previous neighbor with high comp
                        self.cube_competence[cube_idx] = min(self.cube_competence[cube_idx] + 1, self.max_per_cube)
                        break
        normalized_competence = np.interp(self.cube_competence[cube_idx], (0, self.max_per_cube), (0, 1))
        # if self.noise >= 0.0:
        #     normalized_competence = np.clip(normalized_competence + np.random.normal(0,self.noise), 0, 1)
        return normalized_competence


# Controller functions for various teacher algorithms
def test_riac(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    # Init teacher
    task_generator = RIAC(np.array([0.0] * nb_dims), np.array([1.0]*nb_dims), params=params)

    # Init book keeping
    all_boxes = []
    iterations = []
    alps = []
    rewards = []
    scores = []

    # Launch run
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i, scores[-1]))
        task = task_generator.sample_task()
        reward = env.episode(task)
        split, _ = task_generator.update(np.array(task), reward)

        # Book keeping if RIAC performed a new split
        if split and gif:
            boxes = task_generator.regions_bounds
            alp = task_generator.regions_alp
            alps.append(copy.copy(alp))
            iterations.append(i)
            all_boxes.append(copy.copy(boxes))
        rewards.append(reward)

    if gif and nb_dims==2:
        print('Creating gif...')
        region_plot_gif(all_boxes, alps, iterations, task_generator.sampled_tasks,
                        gifname='riac_'+str(time.time()), ep_len=[1]*nb_episodes, rewards=rewards, gifdir='toy_env_gifs/')
        print('Done (see graphics/toy_env_gifs/ folder)')
    return scores

def test_alpgmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    # Init teacher
    task_generator = ALPGMM([0] * nb_dims, [1] * nb_dims, params=params)

    # Init book keeping
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'tasks_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}

    # Launch run
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i, scores[-1]))

        # Book keeping if ALP-GMM updated its GMM
        if i>100 and (i % task_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(task_generator.gmm.weights_.copy())
            bk['covariances'].append(task_generator.gmm.covariances_.copy())
            bk['means'].append(task_generator.gmm.means_.copy())
            bk['tasks_lps'] = task_generator.tasks_alps
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['comp_grids'].append(env.cube_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())

        task = task_generator.sample_task()
        reward = env.episode(task)
        task_generator.update(np.array(task), reward)
        rewards.append(reward)

    if gif and nb_dims==2:
        print('Creating gif...')
        gmm_plot_gif(bk, gifname='alpgmm_'+str(time.time()), gifdir='toy_env_gifs/')
        print('Done (see graphics/toy_env_gifs/ folder)')
    return scores

def test_covar_gmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    # Init teacher
    task_generator = CovarGMM([0] * nb_dims, [1] * nb_dims, params=params)

    # Init book keeping
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'tasks_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}

    # Launch run
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i,scores[-1]))

        # Book keeping if Covar-GMM updated its GMM
        if i>100 and (i % task_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(task_generator.gmm.weights_.copy())
            bk['covariances'].append(task_generator.gmm.covariances_.copy())
            bk['means'].append(task_generator.gmm.means_.copy())
            bk['tasks_lps'] = task_generator.tasks_times_rewards
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['comp_grids'].append(env.cube_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())
        task = task_generator.sample_task()
        reward = env.episode(task)
        task_generator.update(np.array(task), reward)
        rewards.append(reward)

    if gif and nb_dims==2:
        print('Creating gif...')
        gmm_plot_gif(bk, gifname='covargmm_'+str(time.time()), gifdir='toy_env_gifs/')
        print('Done (see graphics/toy_env_gifs/ folder)')
    return scores

def test_random(env, nb_episodes, nb_dims=2, gif=False, score_step=1000, verbose=True, params={}):
    scores = []
    
    # Init Book-keeping
    gif_step_size=250  # to match ALP-GMM and Covar-GMM gif style
    bk = {'comp_grids': [], 'comp_xs': [], 'comp_ys': [], 'tasks':[]}
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i,scores[-1]))
            
        # Book-keeping
        if i > 100 and (i % gif_step_size) == 0 and (gif is True):
            if nb_dims == 2:
                bk['comp_grids'].append(env.cube_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())

        p = np.random.random(nb_dims)
        env.episode(p)
        
    if gif and nb_dims==2:
        bk['tasks'] = env.params
        print('Creating gif...')
        random_plot_gif(bk, gifname='random_'+str(time.time()), gifdir='toy_env_gifs/')
        print('Done (see graphics/toy_env_gifs/ folder)')
    return scores



if __name__=="__main__":
    nb_episodes = 20000
    nb_dims = 2
    nb_cubes = 10
    score_step = 1000
    env = ToyEnv(nb_dims=nb_dims, nb_cubes=nb_cubes)
    all_scores = []
    colors = ['grey','orange','blue','green']
    labels = ['Random', 'RIAC', 'ALP-GMM', 'Covar-GMM']
    print('Testing Random...')
    all_scores.append(test_random(env, nb_episodes, gif=True, nb_dims=nb_dims, score_step=score_step, verbose=True))
    env.reset()
    print('Testing RIAC...')
    all_scores.append(test_riac(env, nb_episodes, gif=True, nb_dims=nb_dims, score_step=score_step, verbose=True))
    env.reset()
    print('Testing ALP-GMM...')
    all_scores.append(test_alpgmm(env, nb_episodes, gif=True, nb_dims=nb_dims, score_step=score_step, verbose=True))
    env.reset()
    print('Testing Covar-GMM...')
    all_scores.append(test_covar_gmm(env, nb_episodes, gif=True, nb_dims=nb_dims, score_step=score_step, verbose=True))


    # Plot evolution of % of mastered hypercubes
    episodes = np.arange(0, nb_episodes + score_step, score_step) / score_step
    ax = plt.gca()
    for scores, color, label in zip(all_scores, colors, labels):
        ax.plot(episodes, scores, color=color, label=label, linewidth=5)
    ax.legend()
    ax.set_xlabel('Episodes (x1000)', fontsize=20)
    ax.set_ylabel('% Mastered cubes', fontsize=20)
    ax.set_xlim(xmin=0, xmax=nb_episodes / score_step)
    ax.set_ylim(ymin=0, ymax=100)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()
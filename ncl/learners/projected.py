"""Projection-based learners
"""

import numpy as onp
import json
import jax
from jax import grad, jit, vmap, lax, random
import jax.numpy as np
from . import tools, fisher
from .. import rnns
from ..tasks import TaskSet
from .base import AbstractLearner
import abc


def update_l(p, m, projs, lr):
    return p - lr * (projs['L'] @ m)


def update_r(p, m, projs, lr):
    return p - lr * (m @ projs['R'])


def update_both(p, m, projs, lr):
    return p - lr * (projs['L'] @ m @ projs['R'])


update_mapping = {'left': update_l, 'right': update_r, 'both': update_both}


class AbstractOrthogonal(AbstractLearner, metaclass=abc.ABCMeta):
    def __init__(self,
                 mode,
                 hp,
                 save_dir: str,
                 taskset: TaskSet,
                 phi=jax.nn.relu,
                 projection_alpha=1e-3,
                 l2_w=1e-5,
                 l2_h=1e-7):

        super().__init__(hp, save_dir, taskset, phi=phi)
        self.projection_alpha = projection_alpha
        self.hp['projection_alpha'] = projection_alpha
        self.l2_w = l2_w
        self.l2_h = l2_h
        self.hp['l2_w'] = l2_w
        self.hp['l2_h'] = l2_h
        self.mode = mode
        self._update = update_mapping[mode]
        self.compute_covs = fisher.orthogonal(hp, self.vrnn)

        tools.save_pickled_proj(None, self.save_dir, init=True)
        tools.save_all_states(None, self.save_dir, init=True)

    def __call__(self, rule_set):
        hp = self.hp
        with open(self.save_dir + '/hp.json', 'w') as f:
            json.dump(hp, f)
        key, subkey = random.split(jax.random.PRNGKey(hp['seed']))
        params0 = self.init_rnn_params(subkey)
        state = self.init_state(params0)
        subkeys = random.split(key, len(rule_set))
        step = 0
        covs = self.init_covs()
        projs = None
        tools.save_all_states(self.get_params(state), self.save_dir)

        #learn each task sequentially
        for rule_i, (subkey, rule) in enumerate(zip(subkeys, rule_set)):
            learner_update = self.init_learner_update(
                state, projs)  #define update step etc.
            subkey1, subkey2 = random.split(subkey)
            state, step = self.learn(subkey1, state, learner_update, rule,
                                     rule_set, step)  #train on task i
            #update Fisher matrices matrices with data from most recent task
            covs = self.update_covs(subkey2, state, covs, rule, rule_i)
            projs = self.compute_projs(covs)  #compute projection matrices
            tools.save_all_states(self.get_params(state), self.save_dir)

    def init_covs(self):
        """we do not incorporate explicit priors"""
        hp = self.hp
        n_h = hp['n_hidden']
        n_i = hp['n_input']
        n_o = hp['n_output']
        n_z = n_h + n_i
        return {
            'z': np.zeros((n_z, n_z)),
            'wz': np.zeros((n_h, n_h)),
            'y': np.zeros((n_o, n_o)),
        }

    def update_covs(self, key, state, covs, rule, rule_i):
        """Updates the covariances of the learner

        Args:
            key: Jax PRNG Key
            state: optimization state
            covs: old covs
            rule (str): current task
            rule_i (int): index of current task
        
        Returns:
            covs: updated covs
        """
        # initialize covs if it is None
        hp = self.hp
        #generate trials for computing cov matrices
        trial = self.taskset.generate_trials(self.rng, hp, rule,
                                             hp['eval_batch_size'],
                                             'train')  #generate trial input

        key, subkey = random.split(key)
        new_covs = self.compute_covs(key, trial, self.get_params(state))
        #online update of the full covariance; average over tasks
        beta = 1. / (rule_i + 1)
        sigma_z = tools.update_cov(covs['z'], new_covs['z'], beta)
        sigma_y = tools.update_cov(covs['y'], new_covs['y'], beta)
        sigma_wz = tools.update_cov(covs['wz'], new_covs['wz'], beta)
        #sigma_z = tools.update_cov(covs['z'], sigma_z_, beta)
        #sigma_y = tools.update_cov(covs['y'], sigma_y_, beta)
        #sigma_wz = tools.update_cov(covs['wz'], sigma_wz_, beta)

        return {
            'z': sigma_z,
            'wz': sigma_wz,
            'y': sigma_y,
        }

    def compute_projs(self, covs):
        """Compute projections base on the covs 

        Args:
            covs: the covariances given by learnt tasks
        
        Returns:
            The projection matrices used in learning a new task.
        """
        hp = self.hp
        alpha = self.projection_alpha
        sigma_h = covs['z'][0:hp['n_hidden'], 0:hp['n_hidden']]
        p_h = tools.reginv(sigma_h, alpha) * alpha
        p_y = tools.reginv(covs['y'], alpha) * alpha
        p_z = tools.reginv(covs['z'], alpha) * alpha
        p_wz = tools.reginv(covs['wz'], alpha) * alpha
        projs = {
            'w': {
                'L': p_z,
                'R': p_wz
            },
            'w_out': {
                'L': p_h,
                'R': p_y
            },
        }

        tools.save_pickled_proj(projs, self.save_dir)

        return projs

    def init_learner_update(self, state, projs):
        """Initialize the learner update function at the beginning of learning each new task

        Args:
            state: current optimization state (params, momentum)
            projs: the prior given by learnt tasks (None by default)

        Returns:
            The ``learner_update`` function for learning the new task.
        """
        l2_w = self.hp['l2_w']
        l2_h = self.hp['l2_h']

        def reg_w(params):
            reg_w_out = np.sum(params['w_out']**2)
            reg_w = np.sum(params['w']**2)
            return 0.5 * l2_w * (reg_w_out + reg_w)

        def reg_h(h):
            return 0.5 * l2_h * np.sum(h**2)

        def cost(params, trial, key):
            x = trial['x']
            etas = rnns.process_noise(
                key, self.hp, (x.shape[0], x.shape[1], self.hp['n_hidden']))
            ypreds, h = self.vrnn(x, etas, trial['mask'],
                                  params)  #output and hidden states of RNN
            lsq = self.vnegloglik(ypreds, trial['y'], trial['c_mask'],
                                  trial['mask'], trial['Rinv'])
            reg_h_val = lax.cond(l2_h > 0,
                                 lambda _: reg_h(h),
                                 lambda _: 0.,
                                 operand=None)
            reg_w_val = lax.cond(l2_w > 0,
                                 lambda _: reg_w(params),
                                 lambda _: 0.,
                                 operand=None)

            total = np.mean(lsq) + reg_h_val + reg_w_val
            return total

        def update(key, state, trial):
            params, ms = state
            g = grad(cost)(params, trial, key)
            new_ms = {k: (self.mass * ms[k]) + g[k] for k in ms}

            if projs is None:
                lr = self.learning_rate
                new_params = {k: params[k] - lr * new_ms[k] for k in params}
            else:
                lr = self.learning_rate
                new_params = {
                    k: self._update(params[k], new_ms[k], projs[k], lr)
                    for k in params
                }
            return new_params, new_ms

        return jit(update)


class LOrthogonal(AbstractOrthogonal):
    name = "lorthogonal"

    def __init__(
        self,
        hp,
        save_dir: str,
        taskset: TaskSet,
        phi=jax.nn.relu,
        projection_alpha=1e-3,
        l2_w=1e-5,
        l2_h=1e-7,
    ):

        super().__init__(
            'left',
            hp,
            save_dir,
            taskset,
            phi=phi,
            projection_alpha=projection_alpha,
            l2_w=l2_w,
            l2_h=l2_h,
        )


class ROrthogonal(AbstractOrthogonal):
    name = "rorthogonal"

    def __init__(self,
                 hp,
                 save_dir: str,
                 taskset: TaskSet,
                 phi=jax.nn.relu,
                 projection_alpha=1e-3,
                 l2_w=1e-5,
                 l2_h=1e-7):

        super().__init__('right',
                         hp,
                         save_dir,
                         taskset,
                         phi=phi,
                         projection_alpha=projection_alpha,
                         l2_w=l2_w,
                         l2_h=l2_h)


class Orthogonal(AbstractOrthogonal):
    name = "orthogonal"

    def __init__(self,
                 hp,
                 save_dir: str,
                 taskset: TaskSet,
                 phi=jax.nn.relu,
                 projection_alpha=1e-3,
                 l2_w=1e-5,
                 l2_h=1e-7):

        super().__init__('both',
                         hp,
                         save_dir,
                         taskset,
                         phi=phi,
                         projection_alpha=projection_alpha,
                         l2_w=l2_w,
                         l2_h=l2_h)

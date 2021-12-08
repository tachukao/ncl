"""Online Laplace-based learners
"""

import numpy as onp
import json
import jax
from jax import grad, jit, vmap, lax, random
import jax.numpy as np
from . import tools, fisher
from ..tasks import TaskSet
from .. import rnns
from .base import AbstractLearner
import abc
import pickle


class Laplace(AbstractLearner, metaclass=abc.ABCMeta):
    """Laplace Learner class

    Args:
        hp: hyperparameter dictionary
        save_dir (str): directory to save the results
        taskset (Taskset): the taskset that we will be learning from (usually only learning a subset of "rules" from the taskset)
        phi: nonlinearity of the recurrent network
    """
    def __init__(self, hp, save_dir: str, taskset: TaskSet, phi=jax.nn.relu):
        super().__init__(hp, save_dir, taskset, phi=phi)
        tools.save_pickled_fisher(None, self.save_dir, init=True)
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
        priors = self.init_priors()  #initialize non-zero priors!

        #learn each task sequentially
        tools.save_all_states(state, self.save_dir)
        for rule_i, (subkey, rule) in enumerate(zip(subkeys, rule_set)):
            learner_update = self.init_learner_update(state, priors, rule_i)
            subkey1, subkey2 = random.split(subkey)
            state, step = self.learn(subkey1, state, learner_update, rule,
                                     rule_set, step)
            priors = self.update_priors(subkey2, state, priors, rule)
            tools.save_all_states(state, self.save_dir)

        params = self.get_params(state)
        pickle.dump(params, open(self.save_dir + '/final_params.pickled',
                                 'wb'))

    @abc.abstractmethod
    def init_priors(self):
        pass

    @abc.abstractmethod
    def update_priors(self, key, state, priors, rule):
        pass


class KFAC(Laplace):
    name = "kfac"

    def __init__(self, hp, save_dir: str, taskset: TaskSet, phi=jax.nn.relu):
        super().__init__(hp, save_dir, taskset, phi=phi)
        self.kfac_fisher = fisher.kfac(hp, self.vrnn, self.rnn_aux,
                                       self.taskset.negloglik,
                                       self.taskset.sample_target)

    def init_priors(self):
        """Initializes the priors
        """
        hp = self.hp
        priors = {
            'z':
            np.eye(hp['n_input'] + hp['n_hidden']) / np.sqrt(hp['data_size']),
            'r': np.eye(hp['n_hidden']) / np.sqrt(hp['data_size']),
            'hbar': np.eye(hp['n_hidden']) / np.sqrt(hp['data_size']),
            'ybar': np.eye(hp['n_output']) / np.sqrt(hp['data_size'])
        }
        tools.save_pickled_fisher(priors, self.save_dir)
        return priors

    def update_priors(self, key, state, priors, rule):
        """Same as ``update_priors`` of Orthogonal.
        """
        hp = self.hp

        #generate trial input
        trial = self.taskset.generate_trials(self.rng, hp, rule,
                                             hp['eval_batch_size'], 'train')
        params = self.get_params(state)
        key, subkey = random.split(key)
        fishers = self.kfac_fisher(subkey, trial, params)

        tools.save_pickled_fisher(fishers, self.save_dir)

        key, subkey = random.split(key)
        p_rec = fisher.opt_nearest_kf(subkey, 1000, {
            'R': priors['hbar'],
            'L': priors['z']
        }, {
            'R': fishers['hbar'],
            'L': fishers['z']
        })
        key, subkey = random.split(key)
        p_out = fisher.opt_nearest_kf(subkey, 1000, {
            'R': priors['ybar'],
            'L': priors['r']
        }, {
            'R': fishers['ybar'],
            'L': fishers['r']
        })

        new_priors = {
            'z': p_rec['L'],
            'r': p_out['L'],
            'hbar': p_rec['R'],
            'ybar': p_out['R']
        }

        return new_priors

    def init_learner_update(self, state, priors, rule_i):
        """Same as ``init_learner_update`` of Orthogonal.
        """
        params0 = self.get_params(state)
        if rule_i == 0:
            params0 = {k: np.zeros_like(params0[k]) for k in params0}
        hp = self.hp

        fisher = {
            'w': {
                'L': priors['z'],
                'R': priors['hbar']
            },
            'w_out': {
                'L': priors['r'],
                'R': priors['ybar']
            },
        }

        def reg(p, p0, fisher):
            dp = p - p0
            return 0.5 * np.sum((fisher['L'] @ dp @ fisher['R']) * dp)

        def cost(params, trial, key):
            x = trial['x']
            etas = rnns.process_noise(
                key, self.hp, (x.shape[0], x.shape[1], self.hp['n_hidden']))
            ypreds, _ = self.vrnn(x, etas, trial['mask'], params)
            lik = self.vnegloglik(ypreds, trial['y'], trial['c_mask'],
                                  trial['mask'], trial['Rinv'])
            # reg_w, reg_w_out
            regs = {k: reg(params[k], params0[k], fisher[k]) for k in params}
            reg_term = regs['w'] + regs['w_out']
            return np.mean(lik) + (self.hp['lambda'] * reg_term)

        def update(key, state, trial):
            params, ms = state
            g = grad(cost)(params, trial, key)
            new_ms = {k: (self.mass * ms[k]) + g[k] for k in ms}
            lr = self.learning_rate / (1 + (50 * rule_i))
            new_params = {k: params[k] - lr * new_ms[k] for k in params}
            return new_params, new_ms

        return jit(update)


class KFACAdam(KFAC):
    name = "kfac-adam"

    def __init__(self, hp, save_dir: str, taskset: TaskSet, phi=jax.nn.relu):
        super().__init__(hp, save_dir, taskset, phi=phi)
        self.beta1 = 0.9
        self.beta2 = 0.999

    def init_state(self, params):
        """Initializes the optimization state given initial parameters `params`. 
        Here the optimization state is a tuple (params, momentum)
        """
        m = {k: np.zeros_like(params[k]) for k in params}
        v = {k: np.zeros_like(params[k]) for k in params}
        return params, m, v, 0

    def get_params(self, state):
        """Returns the parameters of the optimization state
        """
        params, _, _, _ = state
        return params

    def init_learner_update(self, state, priors, rule_i):
        """Same as ``init_learner_update`` of Orthogonal.
        """
        params0 = self.get_params(state)
        if rule_i == 0:
            params0 = {k: np.zeros_like(params0[k]) for k in params0}

        fisher = {
            'w': {
                'L': priors['z'],
                'R': priors['hbar']
            },
            'w_out': {
                'L': priors['r'],
                'R': priors['ybar']
            },
        }

        def reg(p, p0, fisher):
            dp = p - p0
            return 0.5 * np.sum((fisher['L'] @ dp @ fisher['R']) * dp)

        def cost(params, trial, key):
            x = trial['x']
            etas = rnns.process_noise(
                key, self.hp, (x.shape[0], x.shape[1], self.hp['n_hidden']))
            ypreds, _ = self.vrnn(x, etas, trial['mask'], params)
            lik = self.vnegloglik(ypreds, trial['y'], trial['c_mask'],
                                  trial['mask'], trial['Rinv'])
            regs = {k: reg(params[k], params0[k], fisher[k]) for k in params}
            reg_term = regs['w'] + regs['w_out']
            return np.mean(lik) + (self.hp['lambda'] * reg_term)

        def update(key, state, trial):
            beta1 = self.beta1
            beta2 = self.beta2
            params, ms, vs, step = state
            g = grad(cost)(params, trial, key)
            new_ms = {k: (beta1 * ms[k]) + ((1 - beta1) * g[k]) for k in ms}
            new_vs = {
                k: (beta2 * vs[k]) + ((1 - beta2) * (g[k]**2))
                for k in vs
            }
            new_ms_hat = {
                k: new_ms[k] / (1 - np.power(beta1, step + 1))
                for k in new_ms
            }
            new_vs_hat = {
                k: new_vs[k] / (1 - np.power(beta2, step + 1))
                for k in new_vs
            }
            new_params = {
                k: params[k] - self.learning_rate *
                (new_ms_hat[k] / ((np.sqrt(new_vs_hat[k]) + 1e-9)))
                for k in params
            }
            return new_params, new_ms, new_vs, step + 1

        return jit(update)


class NCL(KFAC):
    name = "ncl"

    def __init__(self,
                 hp,
                 save_dir: str,
                 taskset: TaskSet,
                 phi=jax.nn.relu,
                 projection_alpha=1e-6):
        super().__init__(hp, save_dir, taskset, phi=phi)
        self.projection_alpha = projection_alpha
        self.hp['projection_alpha'] = projection_alpha

        tools.save_pickled_proj(None, self.save_dir, init=True)

    def compute_projs(self, priors):
        """Same as ``compute_projs`` of Orthogonal.
        """
        hp = self.hp
        alpha = self.projection_alpha

        p_rec = fisher.additive_nearest_kf(
            {
                'R': priors['hbar'],
                'L': priors['z']
            }, {
                'R': alpha * np.eye(priors['hbar'].shape[0]),
                'L': alpha * np.eye(priors['z'].shape[0])
            })
        p_out = fisher.additive_nearest_kf(
            {
                'R': priors['ybar'],
                'L': priors['r']
            }, {
                'R': alpha * np.eye(priors['ybar'].shape[0]),
                'L': alpha * np.eye(priors['r'].shape[0])
            })

        priors = {
            'z': p_rec['L'],
            'r': p_out['L'],
            'hbar': p_rec['R'],
            'ybar': p_out['R']
        }

        p_r = tools.reginv(priors['r'], 1e-9)
        p_z = tools.reginv(priors['z'], 1e-9)
        p_hbar = tools.reginv(priors['hbar'], 1e-9)
        p_ybar = tools.reginv(priors['ybar'], 1e-9)

        projs = {
            'w': {
                'L': p_z,
                'R': p_hbar
            },
            'w_out': {
                'L': p_r,
                'R': p_ybar
            },
        }

        tools.save_pickled_proj(projs, self.save_dir)

        return projs

    def init_learner_update(self, state, priors, rule_i):
        """Same as ``init_learner_update`` of Orthogonal.
        """
        params0 = self.get_params(state)
        projs = self.compute_projs(priors)

        if rule_i == 0:
            params0 = {k: np.zeros_like(params0[k]) for k in params0}

        fishers = {
            'w': {
                'L': priors['z'],
                'R': priors['hbar']
            },
            'w_out': {
                'L': priors['r'],
                'R': priors['ybar']
            },
        }

        def cost(params, trial, key):
            x = trial['x']
            etas = rnns.process_noise(
                key, self.hp, (x.shape[0], x.shape[1], self.hp['n_hidden']))
            ypreds, _ = self.vrnn(x, etas, trial['mask'], params)
            lp = self.vnegloglik(ypreds, trial['y'], trial['c_mask'],
                                 trial['mask'], trial['Rinv'])
            return np.mean(lp)

        def reg_gradient(p, p0, fisher):
            dp = p - p0
            return fisher['L'] @ dp @ fisher['R']

        def _update(p, m, projs):
            pl = projs['L']
            pr = projs['R']
            lr = self.hp['learning_rate'] / self.hp['data_size']
            return p - lr * (pl @ m @ pr)

        def update(key, state, trial):
            params, ms = state
            g_current_task = grad(cost)(params, trial, key)
            g_reg = {
                k: reg_gradient(params[k], params0[k], fishers[k])
                for k in params
            }
            g = {
                k: g_current_task[k] + self.hp['lambda'] * g_reg[k]
                for k in g_current_task
            }

            new_ms = {k: (self.mass * ms[k]) + g[k] for k in ms}

            #always specify the fisher per datapoint and LL per datapoint
            new_params = {
                k: _update(params[k], new_ms[k], projs[k])
                for k in params
            }

            return new_params, new_ms

        return jit(update)


class LaplaceOWM(NCL):
    name = "laplace-owm"

    def __init__(self,
                 hp,
                 save_dir: str,
                 taskset: TaskSet,
                 phi=jax.nn.relu,
                 projection_alpha=1e-6):
        super().__init__(hp,
                         save_dir,
                         taskset,
                         phi=phi,
                         projection_alpha=projection_alpha)
        self.kfac_fisher = fisher.kfac_owm(self.hp, self.vrnn)


class LaplaceDOWM(NCL):
    name = "laplace-dowm"

    def __init__(self,
                 hp,
                 save_dir: str,
                 taskset: TaskSet,
                 phi=jax.nn.relu,
                 projection_alpha=1e-6):
        super().__init__(hp,
                         save_dir,
                         taskset,
                         phi=phi,
                         projection_alpha=projection_alpha)

        self.kfac_fisher = fisher.kfac_dowm(self.hp, self.vrnn)

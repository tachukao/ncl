"""Base Learner Module
"""

import json
import numpy as onp
import jax
from jax import grad, jit, vmap, lax, random
import jax.numpy as np
from .. import tasks
from ..tasks import TaskSet
from . import tools
from .. import rnns
from collections import defaultdict
import abc


class AbstractLearner(metaclass=abc.ABCMeta):
    def __init__(self, hp, save_dir: str, taskset: TaskSet, phi=jax.nn.relu):
        self.hp = hp
        self.rng = onp.random.RandomState(hp['seed'])  # numpy rng
        self.rnn, self.init_rnn_params, self.rnn_aux = rnns.leaky_rnn(hp, phi)
        self.vrnn = vmap(self.rnn, (0, 0, 0, None))
        self.test_log = defaultdict(list)
        self.taskset = taskset
        self.vnegloglik = jit(vmap(taskset.negloglik, (0, 0, 0, 0, None)))
        self.vperformance = vmap(taskset.performance, (0, {
            'y_loc': 0,
            'tdim': 0,
            'x': None,
            'y': None,
            'mask': None,
            'c_mask': None,
            'sigma_mask': None,
            'R': None,
            'Rinv': None,
        }))
        self.train_batch_size = hp['train_batch_size']
        self.eval_batch_size = hp['eval_batch_size']
        self.learning_rate = hp['learning_rate']
        self.mass = hp['mass']
        self.save_dir = save_dir

    @abc.abstractproperty
    def name(self):
        """Name of the learner"""
        pass

    def init_state(self, params):
        """Initializes the optimization state given initial parameters `params`. Here the optimization state is a tuple (params, momentum)
        """
        m = {k: np.zeros_like(params[k]) for k in params}
        return params, m

    def get_params(self, state):
        """Returns the parameters of the optimization state
        """
        params, _ = state
        return params

    def checkpoint(self, state):
        """Saves the model and test log every ``display_every`` steps (defined in ``hp``)
        """

        tools.save_params(self.get_params(state), self.save_dir)
        tools.save_log_json(self.test_log, self.save_dir)
        tools.save_log_csv(self.test_log, self.save_dir)

    def print_log(self, rule_set):
        """Prints the test log as we go along every ``display_every`` iteration (specified in ``hp``)
        """
        step = self.test_log['step'][-1]
        crule = self.test_log['current-rule'][-1]
        negloglik = lambda rule: self.test_log[rule + '-negloglik'][-1]
        perf = lambda rule: self.test_log[rule + '-perf'][-1]
        rulef = lambda rule: f'>{rule}' if rule == crule else rule
        losses = [
            f"{rulef(rule):10s}\t({negloglik(rule):.4f}, {perf(rule):.2f})"
            for rule in rule_set
        ]
        print(
            f"model {self.name} | n_hidden {self.hp['n_hidden']} | step {step:5d} | current {crule}"
        )
        print('\n'.join(losses))

    def evaluate(self, key, state, rule_set, rule, i: int, step: int):
        """Evaluate model performance on test data

        Args:
            key: Jax PRNG Key
            state: optimization state
            rule_set: set of rules that we want to evaluate on
            rule: current rule we are learning
            i (int): number of training steps in learning the current rule
            step (int): number of training steps overall
        """
        self.test_log['step'].append(step)
        self.test_log['iteration'].append(i)
        self.test_log['current-rule'].append(rule)
        params = self.get_params(state)
        subkeys = random.split(key, len(rule_set))
        for subkey, rule in zip(subkeys, rule_set):  #test on each task
            trial = self.taskset.generate_trials(self.rng, self.hp, rule,
                                                 self.eval_batch_size, 'test')
            xs = trial['x']
            etas = rnns.process_noise(
                subkey, self.hp,
                (xs.shape[0], xs.shape[1], self.hp['n_hidden']))
            #output and hidden states of RNN
            ypreds, _ = self.vrnn(xs, etas, trial['mask'], params)
            negloglik = self.vnegloglik(ypreds, trial['y'], trial['c_mask'],
                                        trial['mask'], trial['Rinv'])
            perf = np.mean(self.vperformance(ypreds, trial))

            self.test_log[rule + '-negloglik'].append(float(
                np.mean(negloglik)))
            self.test_log[rule + '-perf'].append(float(perf))

    def learn(self, key, state, learner_update, rule, rule_set, step):
        """Full round of training on a given task

        Args:
            key: Jax PRNG Key
            state: optimization state
            learner_update: function to optimize optimization state
            rule: current task
            rule_set: all tasks that we want to learn (used for evaluation during training)
            step: total training steps accross all tasks thus far
        
        Returns:
            A tuple of (state, step).
        """
        hp = self.hp
        i = 0
        while (i * self.train_batch_size < hp['max_steps']):
            key, subkey = random.split(key)
            #generate training data
            trial = self.taskset.generate_trials(self.rng, hp, rule,
                                                 self.train_batch_size,
                                                 'train')
            # update state
            state = learner_update(subkey, state, trial)
            if step % hp['display_every'] == 0:
                key, subkey = random.split(key)
                self.evaluate(subkey, state, rule_set, rule, i, step)
                self.checkpoint(state)
                self.print_log(rule_set)
                #test performance
            i = i + 1
            step = step + 1
        return state, step


class Base(AbstractLearner):
    """Learner class

    Args:
        hp: hyperparameter dictionary
        save_dir (str): directory to save the results
        taskset (Taskset): the taskset that we will be learning from (usually only learning a subset of "rules" from the taskset)
        phi: nonlinearity of the recurrent network
    """

    name = "base"

    def __init__(self,
                 hp,
                 save_dir: str,
                 taskset: TaskSet,
                 phi=jax.nn.relu,
                 l2_w=1e-5,
                 l2_h=1e-7):
        super().__init__(hp, save_dir, taskset, phi=phi)
        self.l2_w = l2_w
        self.l2_h = l2_h
        self.hp['l2_w'] = l2_w
        self.hp['l2_h'] = l2_h

    def __call__(self, rule_set):
        hp = self.hp
        # save hp
        with open(self.save_dir + '/hp.json', 'w') as f:
            json.dump(hp, f)
        step = 0
        key, subkey = random.split(jax.random.PRNGKey(hp['seed']))
        params0 = self.init_rnn_params(subkey)
        state = self.init_state(params0)
        subkeys = random.split(key, len(rule_set))
        #learn each task sequentially
        for (subkey, rule) in zip(subkeys, rule_set):
            learner_update = self.init_learner_update()
            state, step = self.learn(subkey, state, learner_update, rule,
                                     rule_set, step)

    def init_learner_update(self):
        """Function that initializes the ``self.learner_update`` function at the beginning of training and after learning a rule.
        """
        l2_w = self.hp['l2_w']
        l2_h = self.hp['l2_h']
        mass = self.hp['mass']

        def reg_w(params):
            reg_w_out = np.sum(params['w_out']**2)
            reg_w = np.sum(params['w']**2)
            return 0.5 * l2_w * (reg_w_out + reg_w)

        def reg_h(h):
            return 0.5 * l2_h * np.sum(h**2)

        def cost(params, key, trial):
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

            #norm = self.taskset.fudge * ypreds.shape[1] * ypreds.shape[2]
            total = np.mean(lsq) + reg_h_val + reg_w_val
            return total

        def update(key, state, trial):
            params, ms = state
            g = grad(cost)(params, key, trial)
            new_ms = {k: (self.mass * ms[k]) + g[k] for k in ms}
            new_params = {
                k: params[k] - self.learning_rate * new_ms[k]
                for k in params
            }
            return new_params, new_ms

        return jit(update)

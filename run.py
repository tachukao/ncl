""" Continual learning in RNN example
"""
import os
from absl import app, flags
from pathlib import Path
import numpy as onp
import ncl
from ncl import learners
import jax
import jax.numpy as np

RESULTS_DIR = flags.DEFINE_string('results_dir', None, 'Results directory')
NAME_EXT = flags.DEFINE_string('name_extension', '', 'Name extension')
TRAIN_BATCH_SIZE = flags.DEFINE_integer('train_batch_size', 32, '')
EVAL_BATCH_SIZE = flags.DEFINE_integer('eval_batch_size', 2048, '')
N_HIDDEN = flags.DEFINE_integer('n_hidden', 256, '')
DISPLAY_EVERY = flags.DEFINE_integer(
    'display_every', 500, 'Display and save results every K iterations')
SAVE_EVERY = flags.DEFINE_integer('save_every', 500,
                                  'Save results every K iterations')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 3e-3, 'Learning rate')
LAMBDA = flags.DEFINE_float('lambda', 1., 'Lambda')
DATA_SIZE = flags.DEFINE_float('data_size', None, 'Data size')
MOMENTUM = flags.DEFINE_float('momentum', 0.9, '')
PROJECTION_ALPHA = flags.DEFINE_float('projection_alpha', 1e-3, '')
SEED = flags.DEFINE_integer('seed', 42, 'Random seed')
MAX_STEPS = flags.DEFINE_integer('max_steps', 1_000_000,
                                 'Maximum number of training trials')
L2_W = flags.DEFINE_float('l2_w', '1e-5', 'L2 penalty on the weights')
L2_H = flags.DEFINE_float('l2_h', '1e-7', 'L2 penalty on the hidden activity')
LEARNER = flags.DEFINE_enum('learner', None, learners.available,
                            'Learner choice')
PHI = flags.DEFINE_enum('nonlinearity', 'relu', ['relu', 'tanh'],
                        "Nonlinearity.")

TASK = flags.DEFINE_string('task', 'ryang', "Task.")
HEAD = flags.DEFINE_enum('head', 'single', ['multi', 'single'], "Head.")


def main(argv):

    phi = ncl.rnns.phi_map[PHI.value]
    task = ncl.tasks.task_map[TASK.value.split('_perm')[0]]
    ruleset = ncl.tasks.get_ruleset(TASK.value)
    hp = task.default_hp(N_HIDDEN.value,
                         TRAIN_BATCH_SIZE.value,
                         EVAL_BATCH_SIZE.value,
                         SEED.value,
                         MAX_STEPS.value,
                         display_every=DISPLAY_EVERY.value,
                         save_every=SAVE_EVERY.value,
                         head=HEAD.value,
                         learning_rate=LEARNING_RATE.value,
                         ruleset=ruleset,
                         lambd=LAMBDA.value,
                         data_size=DATA_SIZE.value)
    print(hp)

    save_dir = os.path.join(RESULTS_DIR.value, TASK.value, LEARNER.value,
                            str(SEED.value), PHI.value, NAME_EXT.value)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    learner_mapping = ncl.learners.build_learner_mapping(
        task, save_dir, L2_W.value, L2_H.value, PROJECTION_ALPHA.value, phi)
    learner = learner_mapping[LEARNER.value](hp, task)

    learner(ruleset)


if __name__ == '__main__':
    flags.mark_flag_as_required('results_dir')
    flags.mark_flag_as_required('learner')
    app.run(main)

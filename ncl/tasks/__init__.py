from . import ryang
from . import smnist
from .taskset import TaskSet
import string
import pickle
import os
from absl import app, flags

__all__ = ['Task', 'ryang', 'smnist', 'task_map', 'ruleset_map']

ryang = TaskSet(ryang.default_hp, ryang.generate_trials, ryang.negloglik,
                ryang.fudge, ryang.sample_target, ryang.performance)
smnist = TaskSet(smnist.default_hp, smnist.generate_trials, smnist.negloglik,
                 smnist.fudge, smnist.sample_target, smnist.performance)

task_map = {'ryang': ryang, 'smnist': smnist}

ruleset_map = {
    'ryang': ['fdgo', 'fdanti', 'delaygo', 'delayanti', 'dm1', 'dm2'],
    'smnist': [
        '2_3',
        '4_5',
        '1_7',
        '8_9',
        '0_6',
        '3xy_2xy',
        '5xy_4xy',
        '7xy_1xy',
        '9xy_8xy',
        '6xy_0xy',
        '2ud_3ud',
        '4ud_5ud',
        '1ud_7ud',
        '8ud_9ud',
        '0ud_6ud',
    ],
}


def get_ruleset(task):
    if 'smnist_perm' in task:
        smnist_perms = pickle.load(
            open(
                os.path.join(flags.FLAGS.data_dir,
                             'data/smnist_permutations.p'), 'rb'))
        ind = int(task.split('perm')[-1])
        return smnist_perms[ind]
    else:
        return ruleset_map[flags.FLAGS.task]

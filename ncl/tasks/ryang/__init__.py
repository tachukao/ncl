""" Stimulus-response task adapted from https://github.com/gyyang/multitask
"""

from .ryang import (generate_trials, default_hp, performance, negloglik,
                    sample_target, fudge)

__all__ = [
    'generate_trials', 'default_hp', 'performance', 'negloglik',
    'sample_target', 'fudge'
]

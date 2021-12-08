import numpy as onp
import jax
import jax.numpy as np
from jax import jit
import six
from . import rules
from .delaygoanti import delaygo, delayanti
from .fdgoanti import fdgo, fdanti
from .dm import dm1, dm2

fudge = 1. / 300.

rule_mapping = {
    'fdgo': fdgo,
    'delaygo': delaygo,
    'fdanti': fdanti,
    'delayanti': delayanti,
    'dm1': dm1,
    'dm2': dm2,
}

rule_name = {
    'fdgo': 'Go',
    'fdanti': 'Anti',
    'delaygo': 'Dly Go',
    'delayanti': 'Dly Anti',
    'dm1': 'DM 1',
    'dm2': 'DM 2',
}


def generate(rule, rng, hp, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        rule: str, the rule for this batch
        hp: dictionary of hyperparameters
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    config = hp
    trial = rule_mapping[rule](rng, config, **kwargs)

    # Add rule input to every task
    if 'rule_on' in kwargs:
        rule_on = kwargs['rule_on']
    else:  # default behavior
        rule_on = None
    if 'rule_off' in kwargs:
        rule_off = kwargs['rule_off']
    else:  # default behavior
        rule_off = None

    # overwrite current rule for input
    if 'replace_rule' in kwargs:
        rule = kwargs['replace_rule']

    if isinstance(rule, six.string_types):
        # rule is not iterable
        # Expand to list
        if 'rule_strength' in kwargs:
            rule_strength = [kwargs['rule_strength']]
        else:
            rule_strength = [1.]
        rule = [rule]

    else:
        if 'rule_strength' in kwargs:
            rule_strength = kwargs['rule_strength']
        else:
            rule_strength = [1.] * len(rule)

    for r, s in zip(rule, rule_strength):
        trial.add_rule(r, on=rule_on, off=rule_off, strength=s)

    if noise_on:
        trial.add_x_noise()

    trial.apply_mask()

    return trial


def default_hp(n_hidden,
               train_batch_size,
               eval_batch_size,
               seed=42,
               max_steps=1_000_000,
               display_every=1000,
               save_every=1000,
               learning_rate=1e-3,
               mass=0.9,
               lambd=1.,
               data_size=None,
               **kwargs):
    '''Get a default hp. '''
    ruleset = 'all'
    num_ring = rules.get_num_ring(ruleset)
    n_rule = rules.get_num_rule(ruleset)

    n_eachring = 2
    n_input = 1 + num_ring * n_eachring + n_rule
    n_output = n_eachring + 1
    hp = {
        # train batch_size
        'train_batch_size': train_batch_size,
        # eval batch_size
        'eval_batch_size': eval_batch_size,
        # input type: normal, multi
        'in_type': 'normal',
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'tau': 0.2,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.1,
        # number of units each ring
        'n_eachring': n_eachring,
        # number of rings
        'num_ring': num_ring,
        # number of rules
        'n_rule': n_rule,
        # first input index for rule units
        'rule_start': 1 + num_ring * n_eachring,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of hidden units
        'n_hidden': n_hidden,
        # ruleset
        'ruleset': ruleset,
        # max steps
        'max_steps': max_steps,
        # random seed
        'seed': seed,
        # display_every
        'display_every': display_every,
        # save_every
        'save_every': save_every,
        # delay_fac
        'delay_fac': 1,
        # data size
        'data_size': (1e6 if data_size is None else data_size),
        # learning_rate
        'learning_rate': learning_rate,
        # mass
        'mass': mass,
        #regularization strength c.f. EWC (the proper Bayesian treatment has lambda=1)
        'lambda': lambd,
    }

    return hp


generate_mapping = {'train': generate, 'test': generate}


def generate_trials(rng, hp, rule, batch_size, mode):
    #generate a trial class instance
    trial = generate_mapping[mode](rule,
                                   rng,
                                   hp,
                                   noise_on=True,
                                   batch_size=batch_size)

    c_mask = trial.c_mask.reshape(trial.y.shape)
    sigma_mask = onp.zeros_like(c_mask)
    sigma_mask[c_mask >= 1e-5] = 1
    return {
        "x": trial.x,
        "y": trial.y,
        "c_mask": c_mask,
        "sigma_mask": sigma_mask,
        "y_loc": trial.y_loc,
        "tdim": trial.tdim,
        "mask": trial.mask,
        "R": np.eye(hp['n_output']),
        "Rinv": np.eye(hp['n_output'])
    }


@jit
def popvec(y):
    """
    Code copied and modified from Dunker et al., 2020

    Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """

    loc = np.arctan2(y[0], y[1])
    return np.mod(loc, 2 * onp.pi)


def performance(y_hat, trial):
    """
    Code copied and modified from Dunker et al., 2020

    Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time)
      Ts: length

    Returns:
      perf: Numpy array (Batch,)
    """
    y_loc = trial['y_loc']
    Ts = trial['tdim']

    y_loc = y_loc[Ts - 1]
    y_hat = y_hat[Ts - 1, :]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))

    corr_loc = dist < 0.1 * np.pi  ###this is 0.1 in D&D, 0.2 in ryang

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    return perf


@jit
def negloglik(ypreds, ytarget, c_mask, mask, Rinv):
    deltas = (ypreds - ytarget) * mask[..., None] * c_mask
    LL = 0.5 * np.sum(deltas**2)
    return fudge * LL


@jit
def sample_target(key, y, c_mask, sigma_mask, mask, R):
    sigma = sigma_mask / (1e-12 + c_mask * np.sqrt(fudge))
    ytarget = y + sigma * jax.random.normal(key, y.shape)
    return mask[..., None] * ytarget

import numpy as np
from .trial import Trial


def fdgo_(rng, config, anti_response, **kwargs):
    '''
    Go with inhibitory control. Important difference with Go task is that
    the stimulus is presented from the beginning.

    Fixate whenever fixation point is shown,
    A stimulus will be shown from the beginning
    And output should saccade to the stimulus location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimulus is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimulus location

    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    batch_size = kwargs['batch_size']
    delay_fac = config['delay_fac']
    # each batch consists of sequences of equal length
    # A list of locations of fixation points and fixation off time

    # A list of locations of stimulus (they are always on)
    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_mod = rng.choice([1, 2], size=(batch_size))
    stim_ons = (rng.uniform(300, 700, size=(batch_size)) / dt).astype(int)

    fix_offs = stim_ons + delay_fac * (
        rng.uniform(500, 1500, size=(batch_size)) / dt).astype(int)
    tdim = fix_offs + (rng.uniform(300, 700, size=(batch_size)) /
                       dt).astype(int)
    # 20190510

    # time to check the saccade location
    check_ons = fix_offs + int(100 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(rng, config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'stim1': (stim_ons, fix_offs),
        'go1': (fix_offs, None)
    }

    return trial


def fdgo(rng, config, **kwargs):
    return fdgo_(rng, config, False, **kwargs)


def fdanti(rng, config, **kwargs):
    return fdgo_(rng, config, True, **kwargs)

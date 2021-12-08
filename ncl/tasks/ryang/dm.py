import numpy as np
from .trial import Trial


def _dm(rng, config, stim_mod, **kwargs):
    """
    Fixate whenever fixation point is shown.
    Two stimuluss are shown, saccade to the one with higher intensity
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    """

    dt = config['dt']
    batch_size = kwargs['batch_size']

    # A list of locations of stimuluss (they are always on)
    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi,
                            (batch_size, )) * rng.choice([-1, 1],
                                                         (batch_size, ))
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size, ))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    # Target strengths
    stims_mean = rng.uniform(0.8, 1.2, (batch_size, ))
    # stims_diff = rng.uniform(0.01,0.2,(batch_size,))
    # stims_diff = rng.choice([0.02, 0.04, 0.08], (batch_size,)) # Encourage integration
    # stims_coh  = rng.choice([0.16, 0.32, 0.64], (batch_size,))

    stim_coh_range = np.random.uniform(0.005, 0.8, (100, ))  #20190805
    stim_coh_range = np.random.uniform(0.4, 0.8, (100, ))  #20190805

    if ('easy_task' in config) and config['easy_task']:
        # stim_coh_range = np.array([0.1, 0.2, 0.4, 0.8])
        stim_coh_range *= 10

    stims_coh = rng.choice(stim_coh_range, (batch_size, ))
    stims_sign = rng.choice([1, -1], (batch_size, ))

    stim1_strengths = stims_mean + stims_coh * stims_sign
    stim2_strengths = stims_mean - stims_coh * stims_sign

    # Time of stimuluss on/off
    stim_ons = (rng.uniform(100, 400, size=(batch_size)) / dt).astype(int)
    #stim_ons = (np.ones(batch_size) * stim_on).astype(int)
    # stim_dur = int(rng.uniform(300,1500)/dt)
    stim_dur = (rng.uniform(600, 1400, size=(batch_size)) / dt).astype(int)
    fix_offs = (stim_ons + stim_dur).astype(int)
    # each batch consists of sequences of equal length
    tdim = stim_ons + stim_dur + (rng.uniform(300, 700, size=(batch_size)) /
                                  dt).astype(int)

    # time to check the saccade location
    check_ons = fix_offs + int(100 / dt)
    stim_mods = (stim_mod * np.ones(batch_size)).astype(int)

    trial = Trial(rng, config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim',
              stim1_locs,
              ons=stim_ons,
              offs=fix_offs,
              strengths=stim1_strengths,
              mods=stim_mods)
    trial.add('stim',
              stim2_locs,
              ons=stim_ons,
              offs=fix_offs,
              strengths=stim2_strengths,
              mods=stim_mods)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [
        stim1_locs[i] if
        (stim1_strengths[i] > stim2_strengths[i]) else stim2_locs[i]
        for i in range(batch_size)
    ]
    trial.add('out', stim_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'stim1': (stim_ons, fix_offs),
        'go1': (fix_offs, None)
    }

    return trial


def dm1(rng, config, **kwargs):
    return _dm(rng, config, 1, **kwargs)


def dm2(rng, config, **kwargs):
    return _dm(rng, config, 2, **kwargs)

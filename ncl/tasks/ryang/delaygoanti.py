import numpy as np
from .trial import Trial


def delaygo_(rng, config, anti_response, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    saccade to the location of the previously shown stimulus
    whenever the fixation point is off
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimulus is shown between (stim_on, stim_off)

    The output should be fixation location for (0, fix_off)
    and the stimulus location for (fix_off, T)

    Optional parameters:
    :param batch_size: Batch size 
    :param tdim: dimension of time 
    :param param: a dictionary of parameters 
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    batch_size = kwargs['batch_size']
    delay_fac = config['delay_fac']

    # A list of locations of stimuluss and on/off time
    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_ons = (rng.uniform(300, 700, size=(batch_size)) / dt).astype(
        int)  #  int(rng.choice([300, 500, 700])/dt) #dec 19th 2018
    stim_offs = stim_ons + (
        rng.uniform(200, 600, size=(batch_size)) / dt).astype(
            int)  #int(rng.choice([200, 400, 600])/dt) # dec 14 2018
    fix_offs = stim_offs + delay_fac * (
        rng.uniform(200, 1600, size=(batch_size)) / dt).astype(
            int)  #int(rng.choice([200, 400, 800, 1600])/dt) # dec 14 2018
    tdim = fix_offs + (rng.uniform(300, 700, size=(batch_size)) / dt).astype(
        int)  # 20190510
    stim_mod = rng.choice([1, 2], size=(batch_size))

    check_ons = fix_offs + int(100 / dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs + np.pi) % (2 * np.pi)

    trial = Trial(rng, config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'stim1': (stim_ons, stim_offs),
        'delay1': (stim_offs, fix_offs),
        'go1': (fix_offs, None)
    }

    return trial


def delaygo(rng, config, **kwargs):
    return delaygo_(rng, config, False, **kwargs)


def delayanti(rng, config, **kwargs):
    return delaygo_(rng, config, True, **kwargs)

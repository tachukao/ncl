import numpy as np
from .rules import get_rule_index


def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


class Trial():
    """Class representing a batch of trials."""
    def __init__(self, rng, config, tdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'  # This should be the default
        self.config = config
        self.rng = rng
        self.dt = self.config['dt']

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.pref = np.arange(0, 2 * np.pi,
                              2 * np.pi / self.n_eachring)  # preferences

        self.batch_size = batch_size
        self.tdim = tdim
        self.maxT = int(np.max(tdim))
        self.mask = np.stack([
            np.concatenate((np.ones((T)), np.zeros((self.maxT - T))))
            for T in self.tdim
        ])

        self.x = np.zeros((batch_size, self.maxT, self.n_input),
                          dtype=self.float_type)
        self.y = np.zeros((batch_size, self.maxT, self.n_output),
                          dtype=self.float_type)
        self.y[:, :, :] = 0.05

        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc = -np.ones((batch_size, self.maxT), dtype=self.float_type)

        self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['tau'])

    def expand(self, var):
        """Expand an int/float to list."""
        if var is None:
            var = [None] * self.batch_size
        return var

    def add(self,
            loc_type,
            locs=None,
            ons=None,
            offs=None,
            strengths=None,
            mods=None):
        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float, strength of input or target output
            mods: int or list, modalities of input or target output
        """

        ons = self.expand(ons)
        offs = self.expand(offs)
        #strengths = self.expand(strengths)
        mods = self.expand(mods)
        locs = self.expand(locs)

        strengths = np.ones(
            self.batch_size) if strengths is None else strengths

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[i, ons[i]:offs[i], 0] = 1
            elif loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                self.x[i, ons[i]: offs[i], 1+(mods[i]-1)*self.n_eachring:1+mods[i]*self.n_eachring] \
                    += self.add_x_loc(locs[i])*strengths[i]
                try:
                    self.stim_locs
                except AttributeError:
                    self.stim_locs = 100 * np.ones(
                        (len(locs), 4), dtype=self.float_type)
                    self.stim_strength = np.zeros((len(locs), 4),
                                                  dtype=self.float_type)

                if self.stim_locs[i, 2 * mods[i] - 2] > 10:
                    self.stim_locs[i, 2 * mods[i] - 2] = locs[i]
                    self.stim_strength[i, 2 * mods[i] - 2] = strengths[i]
                else:
                    self.stim_locs[i, 2 * mods[i] - 1] = locs[i]
                    self.stim_strength[i, 2 * mods[i] - 1] = strengths[i]

            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                self.y[i, ons[i]:offs[i], 0] = 0.8
            elif loc_type == 'out':
                self.y[i, ons[i]:offs[i],
                       1:] += self.add_y_loc(locs[i]) * strengths[i]
                self.y_loc[i, ons[i]:offs[i]] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.rng.randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """

        pre_on = int(100 / self.dt)  # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        c_mask = np.zeros((self.batch_size, self.maxT, self.n_output),
                          dtype=self.float_type)
        for i in range(self.batch_size):
            # Post response periods usually have the same length across tasks
            c_mask[i, post_ons[i]:, :] = 5.
            # Pre-response periods usually have different lengths across tasks
            # To keep cost comparable across tasks
            # Scale the cost mask of the pre-response period by a factor
            c_mask[i, pre_on:pre_offs[i], :] = 1.

        # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
        c_mask[:, :, 0] *= 2.  # Fixation is important

        self.c_mask = c_mask.reshape(
            (self.batch_size, self.maxT, self.n_output)) * self.mask[..., None]

    def add_rule(self, rule, on=None, off=None, strength=1.):
        """Add rule input."""
        if isinstance(rule, int):
            self.x[:, on:off, self.config['rule_start'] + rule] = strength
        else:
            ind_rule = get_rule_index(rule, self.config)
            self.x[:, on:off, ind_rule] = strength

    def add_x_loc(self, x_loc):
        """Input activity given location."""
        return np.array((np.sin(x_loc), np.cos(x_loc)))

    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc - self.pref)  # periodic boundary
        y = np.array((np.sin(y_loc), np.cos(y_loc)))
        return y

    def apply_mask(self):
        mask = self.mask[..., None]
        self.x = self.x * mask
        self.y = self.y * mask
        self.c_mask = self.c_mask * mask
        self.y_loc = self.y_loc * self.mask

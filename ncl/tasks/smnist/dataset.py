import numpy as np
import functools
from absl import flags
import os
import pickle
import copy

DATA_DIR = flags.DEFINE_string('data_dir', None, 'Data directory')


class Dataset():
    def __init__(self):
        pass

    @functools.cached_property
    def data(self):
        data_dir = flags.FLAGS.data_dir
        if data_dir is None:
            raise Exception("Please specify --data_dir.")
        with open(os.path.join(data_dir, 'smnist.pickled'), 'rb') as f:
            return pickle.load(f)

    @functools.cached_property
    def digits(self):
        return [str(digit) for digit in range(10)] + [
            str(digit) + 'xy' for digit in range(10)
        ] + [str(digit) + 'ud' for digit in range(10)]

    @functools.cached_property
    def digit_dict(self):
        return {digit: i for i, digit in enumerate(self.digits)}

    @functools.cached_property
    def max_ind(self):
        return {
            'train':
            {digit: len(self.data['train'][digit])
             for digit in self.digits},
            'test':
            {digit: len(self.data['test'][digit])
             for digit in self.digits}
        }

    @functools.cached_property
    def curr_ind(self):
        return {
            k: {digit: 10000
                for digit in self.digits}  # current index 
            for k in self.max_ind
        }

    @functools.cached_property
    def inds(self):
        return {
            k: {
                digit: np.arange(self.max_ind[k][digit]).astype(int)
                for digit in self.digits
            }
            for k in self.max_ind
        }


def create_generate_trials():
    dataset = Dataset()

    def generate(rng, hp, rule, batch_size, mode='train'):
        """
        rule: str, (e.g. 0_1, 2_3, 4_5, ...)
        if head is multi, the output is 10d with a dimension for each digit
        if not, the output is 2d and all binary classification tasks use the same output channels
        """
        head = hp['smnist_head_type']
        npad = hp['npad']
        #print(dataset.max_ind)

        #### walk through data, then randomize order ###
        curr_ind = dataset.curr_ind[mode]
        inds = dataset.inds[mode]
        max_ind = dataset.max_ind[mode]
        digit_dict = dataset.digit_dict

        #digits to generate data for
        digits = np.array(rule.split('_'))
        #list of number of samples from each category
        nsamps = rng.multinomial(batch_size, [1 / len(digits)] * len(digits))
        nsamps = np.array(
            [min(nsamps[i], max_ind[digits[i]]) for i in range(len(nsamps))])
        batch_size = np.sum(nsamps)

        xs = []
        ys = []
        for idig, digit in enumerate(digits):
            nsamp = nsamps[idig]  #number of samples
            if curr_ind[digit] + nsamp > max_ind[digit]:
                rng.shuffle(inds[digit])  #shuffle data
                curr_ind[digit] = 0  #start over

            digit_inds = inds[digit][curr_ind[digit]:(curr_ind[digit] +
                                                      nsamp)].astype(int)

            curr_ind[digit] += nsamp  #update index

            ### add zeros for the readout period, don't provide the location of the first pixel ###
            newxs = [
                np.concatenate([
                    dataset.data[mode][digit][ind][1:, :],
                    np.zeros((npad, 4))
                ],
                               axis=0) for ind in digit_inds
            ]

            ### add noise ###
            newxs = [
                newx + rng.normal(0, hp['sigma_x'], newx.shape)
                for newx in newxs
            ]

            xs = xs + newxs
            if head == 'multi':
                ys = ys + [digit_dict[digit]] * nsamps[idig]  #target digit
            else:
                ys = ys + [idig] * nsamps[idig]  #target digit index

        Ts = np.array([x.shape[0]
                       for x in xs])  #length of each task including padding
        Tmax = np.amax(Ts)  #largest length needed

        x = np.zeros((batch_size, Tmax, hp['n_input']))
        y = np.zeros((batch_size, Tmax, hp['n_output']))
        c_mask = np.zeros((batch_size, Tmax, hp['n_output']))
        if head == 'single':
            x[..., 4 + hp['task_ind'][rule]] = 1

        for isamp in range(batch_size):
            x[isamp, :Ts[isamp], :4] = xs[isamp]  #input
            y[isamp, (Ts[isamp] - npad):Ts[isamp],
              ys[isamp]] = 1  #ouput is correct digit
            c_mask[isamp, (
                Ts[isamp] - npad
            ):Ts[isamp], :] = 1  #/np.sqrt(npad)  # care about the readout period

        sigma_mask = c_mask

        y_loc = []
        for i in range(batch_size):
            #index of correct classification during classification period
            y_loc.append(np.ones(Tmax) * np.argmax(y[i, Ts[i] - 1]))
        y_loc = np.array(y_loc)

        mask = np.stack(
            [np.concatenate((np.ones((T)), np.zeros((Tmax - T)))) for T in Ts])

        return {
            "x": np.array(x),
            "y": np.array(y),
            "c_mask": np.array(c_mask),
            "sigma_mask": np.array(sigma_mask),
            "y_loc": np.array(y_loc),
            "tdim": np.array(Ts),
            "mask": np.array(mask),
            "R": np.eye(hp['n_output']),
            "Rinv": np.eye(hp['n_output']),
        }

    return generate


if __name__ == '__main__':
    data_to_pickled()

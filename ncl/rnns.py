"""RNN module
"""

import numpy as onp
import jax
from jax import grad, jit, vmap, lax, value_and_grad, random
import jax.numpy as np

phi_map = {
    'linear': lambda x: x,
    'relu': jax.nn.relu,
    'tanh': np.tanh,
    'softplus': jax.nn.softplus
}


def process_noise(key, hp, shape):
    """Generate process noise in the recurrent dynamics.

    Args:
        key:  Jax PRNG key
        hp: hyperparameter dictionary
        shape: shape of the noise
    
    Returns:
        Gaussian noise of dimensions shape with stdev 
        ``np.sqrt(2/tau) * sigma_rec``
    """
    tau = hp['tau']
    sigma_rec = hp['sigma_rec']
    sig = np.sqrt(2 / tau) * sigma_rec
    etas = sig * jax.random.normal(key, shape)
    return etas


def leaky_rnn(hp, phi):
    """Learky RNN generator

    Args:
        hp: hyperparameter dictionary
        phi: RNN nonlinearity

    Returns:
        An (`rnn`, `init_params`, `rnn_aux`) tuple.

        `rnn`
        
            function that simulates the dynamics of the RNN given (inputs ``xs``, noise ``etas``, activity mask ``mask``, parameters ``params``)

        `init_params`
        
            function that initializes the parameters of the RNN

        `rnn_aux`

            auxiliary rnn function that is used for computing the adjoints of the dynamics (``rbars``, ``ybars``).
    """
    tau = hp['tau']
    n_h = hp['n_hidden']
    n_o = hp['n_output']
    n_i = hp['n_input']

    @jit
    def rnn(xs, etas, mask, params):
        w_out = params['w_out']
        w = params['w']
        inps = np.concatenate((xs, etas), -1)
        r0 = np.zeros((n_h, ))

        def f(r, inp):
            x = inp[0:n_i]
            eta = inp[n_i:]
            z = np.concatenate((r, x))
            h = z.dot(w) + eta  # + b_rec
            r_ = ((1 - tau) * r) + (tau * phi(h))
            y = r_.dot(w_out)  # + b_out  #rnn output
            out = np.hstack((y, r_))
            return r_, out

        _, out = lax.scan(f, r0, inps)

        out = out * mask[..., None]
        n_out = params['w_out'].shape[-1]
        y = out[:, 0:n_out]
        r = out[:, n_out:]
        return y, r

    @jit
    def rnn_aux(xs, etas, mask, haux, yaux, params):
        w_out = params['w_out']
        w = params['w']
        r0 = np.zeros((n_h, ))
        inps = np.concatenate((xs, etas, haux, yaux), -1)

        def f(r, inp):
            x = inp[0:n_i]
            eta = inp[n_i:n_i + n_h]
            haux = inp[n_i + n_h:n_i + n_h + n_h]
            yaux = inp[n_i + n_h + n_h:]
            z = np.concatenate((r, x))
            h = z.dot(w) + haux + eta  # + b_rec
            r_ = ((1 - tau) * r) + (tau * phi(h))
            y = r_.dot(w_out) + yaux  # + b_out  #rnn output
            out = np.hstack((y, r_))
            return r_, out

        _, ys = lax.scan(f, r0, inps)

        out = ys * mask[..., None]
        n_out = params['w_out'].shape[-1]
        y = out[:, 0:n_out]
        r = out[:, n_out:]
        return y, r

    def init_params(key):
        subkeys = random.split(key, 3)
        w_out = jax.random.uniform(subkeys[0], (n_h, n_o),
                                   minval=-np.sqrt(1 / (n_o * n_h)),
                                   maxval=np.sqrt(1 / (n_o * n_h)))
        # orthogonal initialization
        w_rec = 0.5 * np.linalg.qr(jax.random.normal(subkeys[1],
                                                     (n_h, n_h)))[0]
        key, subkey = random.split(key)
        w_in = jax.random.normal(subkeys[2], (n_i, n_h)) / np.sqrt(n_i)
        w = np.concatenate((w_rec, w_in), 0)
        params = {'w_out': w_out, 'w': w}
        return params

    return rnn, init_params, rnn_aux

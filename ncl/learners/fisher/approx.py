import numpy as onp
import jax
import jax.numpy as np
from jax import vmap, grad, jit, lax
from .. import tools
from ... import rnns


def diag(hp, vrnn, rnn_aux, negloglik, sample_target):
    n_h = hp['n_hidden']

    def nll(aux, xs, etas, mask, c_mask, ytarget, Rinv, params):
        haux, yaux = aux
        ypreds, _ = rnn_aux(xs, etas, mask, haux, yaux, params)
        return negloglik(ypreds, ytarget, c_mask, mask, Rinv)

    def crosscorr(dw):
        """Cross correlation

        Args:
            dw : dimensions T x m x n
        """
        def inner(carry, y, x):
            carry = carry + (y * x)
            return carry, None

        def outer(carry, x):
            accu = carry
            accu, _ = lax.scan(lambda c, x_: inner(c, x_, x), accu, dw)
            return accu, None

        accu = np.zeros((dw.shape[-2], dw.shape[-1]))
        return lax.scan(outer, accu, dw)[0]

    def f(key, trial, params):
        xs = trial['x']
        mask = trial['mask']
        Ts = trial['tdim']
        c_mask = trial['c_mask']
        sigma_mask = trial['sigma_mask']
        R = trial['R']
        Rinv = trial['Rinv']

        # generate targets
        key, subkey = jax.random.split(key)
        etas = rnns.process_noise(subkey, hp, (xs.shape[0], xs.shape[1], n_h))
        yhat, h = vrnn(xs, etas, mask, params)
        key, subkey = jax.random.split(key)
        ytarget = sample_target(subkey, yhat, c_mask, sigma_mask, mask, R)

        # Compute Fisher
        haux0 = np.zeros((ytarget.shape[0], ytarget.shape[1], n_h))
        yaux0 = np.zeros_like(ytarget)
        hbars, ybars = vmap(grad(nll), ((0, 0), 0, 0, 0, 0, 0, None, None))(
            (haux0, yaux0), xs, etas, mask, c_mask, ytarget, Rinv, params)
        z = np.concatenate((h, xs), -1)  #concatenate hidden state and input
        dw = vmap(vmap(np.outer))(z, hbars)
        dw_out = vmap(vmap(np.outer))(h, ybars)
        return {
            'w': np.mean(vmap(crosscorr)(dw), 0),
            'w_out': np.mean(vmap(crosscorr)(dw_out), 0)
        }

    return jit(f)


def kfac(hp, vrnn, rnn_aux, negloglik, sample_target):
    n_h = hp['n_hidden']

    def nll(aux, xs, etas, mask, c_mask, ytarget, Rinv, params):
        haux, yaux = aux
        ypreds, _ = rnn_aux(xs, etas, mask, haux, yaux, params)
        return negloglik(ypreds, ytarget, c_mask, mask, Rinv)

    def f(key, trial, params):
        xs = trial['x']
        mask = trial['mask']
        Ts = trial['tdim']
        c_mask = trial['c_mask']
        sigma_mask = trial['sigma_mask']
        R = trial['R']
        Rinv = trial['Rinv']

        # generate targets
        key, subkey = jax.random.split(key)
        etas = rnns.process_noise(subkey, hp, (xs.shape[0], xs.shape[1], n_h))
        yhat, h = vrnn(xs, etas, mask, params)
        key, subkey = jax.random.split(key)
        ytarget = sample_target(subkey, yhat, c_mask, sigma_mask, mask, R)

        # Compute Fisher
        haux0 = np.zeros((ytarget.shape[0], ytarget.shape[1], n_h))
        yaux0 = np.zeros_like(ytarget)
        hbars, ybars = vmap(grad(nll), ((0, 0), 0, 0, 0, 0, 0, None, None))(
            (haux0, yaux0), xs, etas, mask, c_mask, ytarget, Rinv, params)
        z = np.concatenate((h, xs), -1)  #concatenate hidden state and input
        sigma_ybar = tools.cov(ybars, Ts, mask)
        sigma_hbar = tools.cov(hbars, Ts, mask)
        sigma_z = tools.cov(z, Ts, mask)  #compute covariance matrix of x & h
        mTs = np.mean(Ts)
        return {
            'ybar': np.sqrt(mTs) * sigma_ybar,
            'hbar': np.sqrt(mTs) * sigma_hbar,
            'z': np.sqrt(mTs) * sigma_z,
            'r': np.sqrt(mTs) * sigma_z[0:n_h, 0:n_h]
        }

    return jit(f)


def kfac_dowm(hp, vrnn):
    n_h = hp['n_hidden']

    def f(key, trial, params):
        xs = trial['x']
        mask = trial['mask']
        Ts = trial['tdim']

        # generate targets
        key, subkey = jax.random.split(key)
        etas = rnns.process_noise(subkey, hp, (xs.shape[0], xs.shape[1], n_h))
        yhat, h = vrnn(xs, etas, mask, params)
        key, subkey = jax.random.split(key)
        z_ = np.concatenate((h, xs), -1)  #concatenate hidden state and input
        sigma_z = tools.cov(z_, Ts, mask)  #compute covariance matrix of x & h
        sigma_y = tools.cov(yhat, Ts, mask)  #compute covariance matrix of y
        sigma_wz = params['w'].T @ sigma_z @ params['w']  #W^T Z^T Z W
        mTs = np.mean(Ts)

        return {
            'ybar': np.sqrt(mTs) * sigma_y,
            'hbar': np.sqrt(mTs) * sigma_wz,
            'z': np.sqrt(mTs) * sigma_z,
            'r': np.sqrt(mTs) * sigma_z[0:n_h, 0:n_h]
        }

    return jit(f)


def kfac_owm(hp, vrnn):
    n_h = hp['n_hidden']

    def f(key, trial, params):
        xs = trial['x']
        mask = trial['mask']
        Ts = trial['tdim']

        # generate targets
        key, subkey = jax.random.split(key)
        etas = rnns.process_noise(subkey, hp, (xs.shape[0], xs.shape[1], n_h))
        yhat, h = vrnn(xs, etas, mask, params)
        key, subkey = jax.random.split(key)
        z_ = np.concatenate((h, xs), -1)  #concatenate hidden state and input
        sigma_z = tools.cov(z_, Ts, mask)  #compute covariance matrix of x & h
        sigma_y = tools.cov(yhat, Ts, mask)  #compute covariance matrix of y
        mTs = np.mean(Ts)

        return {
            'ybar': np.sqrt(mTs) * np.eye(hp['n_output']),
            'hbar': np.sqrt(mTs) * np.eye(n_h),
            'z': np.sqrt(mTs) * sigma_z,
            'r': np.sqrt(mTs) * sigma_z[0:n_h, 0:n_h]
        }

    return jit(f)


def full(hp, vrnn, rnn_aux, negloglik, sample_target):
    n_h = hp['n_hidden']

    def nll(aux, xs, etas, mask, c_mask, ytarget, Rinv, params):
        haux, yaux = aux
        ypreds, _ = rnn_aux(xs, etas, mask, haux, yaux, params)
        return negloglik(ypreds, ytarget, c_mask, mask, Rinv)

    def crosscorr(dw):
        """Cross correlation

        Args:
            dw : dimensions T x p
        """
        T, p = dw.shape

        def inner(carry, y, x):
            carry = carry + np.outer(y, x)
            return carry, None

        def outer(carry, x):
            accu = carry
            accu, _ = lax.scan(lambda c, x_: inner(c, x_, x), accu, dw)
            return accu, None

        accu = np.zeros((p, p))
        return lax.scan(outer, accu, dw)[0]

    def f(key, trial, params):
        xs = trial['x']
        mask = trial['mask']
        Ts = trial['tdim']
        c_mask = trial['c_mask']
        sigma_mask = trial['sigma_mask']
        R = trial['R']
        Rinv = trial['Rinv']

        # generate targets
        key, subkey = jax.random.split(key)
        etas = rnns.process_noise(subkey, hp, (xs.shape[0], xs.shape[1], n_h))
        yhat, h = vrnn(xs, etas, mask, params)
        key, subkey = jax.random.split(key)
        ytarget = sample_target(subkey, yhat, c_mask, sigma_mask, mask, R)

        # Compute Fisher
        haux0 = np.zeros((ytarget.shape[0], ytarget.shape[1], n_h))
        yaux0 = np.zeros_like(ytarget)
        hbars, ybars = vmap(grad(nll), ((0, 0), 0, 0, 0, 0, 0, None, None))(
            (haux0, yaux0), xs, etas, mask, c_mask, ytarget, Rinv, params)
        z = np.concatenate((h, xs), -1)  #concatenate hidden state and input
        dw = vmap(vmap(np.outer))(z, hbars)
        dw = dw.reshape((dw.shape[0], dw.shape[1], -1))  # flatten
        dw_out = vmap(vmap(np.outer))(h, ybars)
        dw_out = dw_out.reshape((dw.shape[0], dw.shape[1], -1))  # flatten
        return {
            'w': np.mean(vmap(crosscorr)(dw), 0),
            'w_out': np.mean(vmap(crosscorr)(dw_out), 0)
        }

    return jit(f)


def orthogonal(hp, vrnn):
    n_h = hp['n_hidden']

    def f(key, trial, params):
        xs = trial['x']
        mask = trial['mask']
        Ts = trial['tdim']

        # generate targets
        key, subkey = jax.random.split(key)
        etas = rnns.process_noise(subkey, hp, (xs.shape[0], xs.shape[1], n_h))
        yhat, h = vrnn(xs, etas, mask, params)
        key, subkey = jax.random.split(key)
        z_ = np.concatenate((h, xs), -1)  #concatenate hidden state and input
        sigma_z = tools.cov(z_, Ts, mask)  #compute covariance matrix of x & h
        sigma_y = tools.cov(yhat, Ts, mask)  #compute covariance matrix of y
        sigma_wz = params['w'].T @ sigma_z @ params['w']  #W^T Z^T Z W
        return {'z': sigma_z, 'y': sigma_y, 'wz': sigma_wz}

    return jit(f)


def owm(hp, vrnn):
    "OWM"
    n_h = hp['n_hidden']

    def f(key, trial, params):
        xs = trial['x']
        mask = trial['mask']
        Ts = trial['tdim']
        c_mask = trial['c_mask']

        # generate targets
        key, subkey = jax.random.split(key)
        etas = rnns.process_noise(subkey, hp, (xs.shape[0], xs.shape[1], n_h))
        yhat, h = vrnn(xs, etas, mask, params)
        key, subkey = jax.random.split(key)
        z_ = np.concatenate((h, xs), -1)  #concatenate hidden state and input
        sigma_z = tools.cov(z_, Ts, mask)  #compute covariance matrix of x & h
        return {'z': sigma_z, 'y': np.eye(hp['n_output']), 'wz': np.eye(n_h)}

    return jit(f)

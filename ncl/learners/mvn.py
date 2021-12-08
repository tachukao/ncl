"""Some useful functions sampling and computing the KL for the Gaussian distributions
"""
from jax import random
import jax.numpy as np
from jax.scipy.linalg import cholesky, solve_triangular

jitter = 1e-8


def tri_kfac_logdet(c):
    dim = {k: c[k].shape[0] for k in c}
    ld1 = np.sum(np.log(np.diag(c['L']))) * dim['R']
    ld2 = np.sum(np.log(np.diag(c['R']))) * dim['L']
    return ld1 + ld2


def kfac_kl(mu1, prec1, mu2, prec2):
    dmu = mu1 - mu2
    u1 = {
        k: cholesky(prec1[k] + jitter * np.eye(prec1[k].shape[0]), lower=False)
        for k in prec1
    }
    u2 = {
        k: cholesky(prec2[k] + jitter * np.eye(prec2[k].shape[0]), lower=False)
        for k in prec2
    }
    kl1 = {tri_blkdiag_logdet(u1[k]) - tri_blkdiag_logdet(u2[k]) for k in u1}
    kl2_ = {
        k: np.sum(np.square(solve_triangular(u1[k].T, u2[k].T)))
        for k in u2
    }
    kl2 = 0.5 * (kl2_['L'] + kl2_['R'])
    kl3 = 0.5 * np.sum((prec2['L'] @ dmu @ prec2['R']) * dmu)
    return kl1 + kl2 + kl3


def kfac_sample(key, mu, prec, scale):
    u = {
        k: cholesky(prec[k] + jitter * np.eye(prec[k].shape[0]), lower=False)
        for k in prec
    }
    eta = random.normal(key, mu.shape) * scale
    return mu + solve_triangular(
        u['R'], solve_triangular(u['L'], eta, lower=False).T, lower=False).T

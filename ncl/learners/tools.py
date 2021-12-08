import csv
import json
import pickle
import jax
from jax import jit, vmap, lax
import jax.numpy as np
from ncl.learners.fisher.approx import full


def save_params(params, result_dir):
    with open('%s/params.p' % result_dir, 'wb') as f:
        pickle.dump(params, f)


def save_all_states(state, result_dir, init=False):
    fname = '%s/all_states.pickled' % result_dir

    if init:
        all_states = []
    else:
        all_states = pickle.load(open(fname, 'rb'))
        all_states.append(state)

    pickle.dump(all_states, open(fname, 'wb'))


def save_log_json(test_log, result_dir):
    # save as json files
    fname = '%s/test_log.json' % result_dir
    with open(fname, 'w') as f:
        json.dump(test_log, f)


def save_pickled_proj(proj, result_dir, init=False):
    # save as json files
    fname = '%s/projs.pickled' % result_dir

    if init:
        projs = {'w': {'L': [], 'R': []}, 'w_out': {'L': [], 'R': []}}
    else:
        projs = pickle.load(open(fname, 'rb'))
        for wkey in ['w', 'w_out']:
            for lrkey in ['L', 'R']:
                projs[wkey][lrkey].append(proj[wkey][lrkey])

    pickle.dump(projs, open(fname, 'wb'))


def save_pickled_fisher(fisher, result_dir, init=False):
    # save as json files
    fname = '%s/fishers.pickled' % result_dir

    if init:
        fishers = {'z': [], 'hbar': [], 'r': [], 'ybar': []}
    else:
        fishers = pickle.load(open(fname, 'rb'))
        for key in ['z', 'hbar', 'r', 'ybar']:
            fishers[key].append(fisher[key])

    pickle.dump(fishers, open(fname, 'wb'))


def save_log_csv(test_log, result_dir):
    fname = '%s/test_log.csv' % result_dir
    with open(fname, 'w') as f:
        fieldnames = test_log.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(test_log["iteration"])):
            writer.writerow({k: test_log[k][i] for k in fieldnames})


@jit
def cov(z, Ts, mask):
    # z has dimensions n_batches, time, dim
    z = z * mask[..., None]
    return (vmap(lambda z: z.T @ z)(z) / (Ts[:, None, None] - 1)).mean(0)


@jit
def update_cov(sigma_past, sigma_new, beta):
    """mean over previous tasks"""
    return ((1 - beta) * sigma_past) + (beta * sigma_new)


@jit
def reginv(x, alpha):
    """Reguarlized symmetric inverse
    """
    x = 0.5 * (x + x.T)
    scale = np.trace(x) / x.shape[0]
    x = x / scale
    evals, v = np.linalg.eigh(x)
    evals_ = 1 / (alpha + (evals * scale))
    return (v * evals_.reshape(1, -1)) @ v.T


def logdet(X, jitter=1e-6):
    d = X.shape[0]
    trX = np.trace(X) / d
    s, ld = np.linalg.slogdet(X / trX + jitter * np.eye(X.shape[0]))
    return s * ld + (d * np.log(trX))


def safe_logdet(X, jitter=1e-6):
    d = X.shape[0]
    scale = d / np.trace(X)
    X = scale * X
    X = X + jitter * np.eye(d)
    evals, _ = np.linalg.eigh(X)
    evals = np.clip(evals, a_min=jitter * 1e-3)
    return np.sum(np.log(evals)) - d * np.log(scale)


def recondition(X):
    """ensure a matrix is symmetric positive semi-definite (zero-out the numerically negative eigenvalues)
    """
    s, u = np.linalg.eigh(X)
    s = np.clip(s, a_min=0.)
    return (u * s.reshape(1, -1)) @ u.T


def safe_inv(X, jitter=1e-8):
    d = X.shape[0]
    recondition(X)
    scale = d / np.trace(X)
    X = scale * X
    evals, v = np.linalg.eigh(X)
    evals = np.clip(evals, a_min=jitter)
    evals_ = 1 / evals
    return scale * (v * evals_.reshape(1, -1)) @ v.T


@jit
def sym_kl(A, B):
    """compute symmetrized (KL[A || B] + KL[B || A]) / 2"""
    d = A.shape[0]
    KL1 = np.trace(safe_inv(B) @ A)
    KL2 = np.trace(safe_inv(A) @ B)
    return np.clip(0.5 * (0.5 * KL1 + 0.5 * KL2) / d, a_min=0.)


@jit
def sym_scaled_kl(A, B, jitter=1e-8):
    """compute symmetrized (KL[A || lambda B] + KL[lambda B || A]) / 2
    for optimal lambda (independent for each direction)"""
    A = recondition(A)
    B = recondition(B)
    d = A.shape[0]
    uA, sA, _ = np.linalg.svd(A, full_matrices=False)
    sA = np.clip(sA, a_min=jitter)
    B = uA.T @ B @ uA
    uB, sB, _ = np.linalg.svd(B, full_matrices=False)
    sB = np.clip(sB, a_min=jitter)
    Z1 = ((uB / sB.reshape((1, -1))) @ uB.T) * sA.reshape((1, -1))
    Z2 = B / sA.reshape((-1, 1))
    KL1 = np.log(np.trace(Z1)) - np.log(d)
    KL2 = np.log(np.trace(Z2)) - np.log(d)
    return np.clip(0.5 * (0.5 * KL1 + 0.5 * KL2), a_min=0.)


def scaled_kl(A, B, jitter=1e-8):
    """compute KL[A || lambda B] for optimal lambda"""
    d = A.shape[0]
    A = recondition(A)
    B = recondition(B)
    uA, sA, _ = np.linalg.svd(A, full_matrices=False)
    sA = np.clip(sA, a_min=jitter)
    B = uA.T @ B @ uA
    sB = np.linalg.svd(B, full_matrices=False, compute_uv=False)
    sB = np.clip(sB, a_min=jitter)
    Z1 = B / sA.reshape((-1, 1))
    KL1 = np.log(np.trace(Z1)) - np.log(d)
    KL2 = (np.sum(np.log(sA)) - np.sum(np.log(sB))) / d
    return np.clip(0.5 * (KL1 + KL2), a_min=0.)


def kl(A, B, jitter=1e-8):
    """compute KL[A || lambda B] for optimal lambda"""
    d = A.shape[0]
    A = recondition(A)
    B = recondition(B)
    uA, sA, _ = np.linalg.svd(A, full_matrices=False)
    sA = np.clip(sA, a_min=jitter)
    B = uA.T @ B @ uA
    sB = np.linalg.svd(B, full_matrices=False, compute_uv=False)
    sB = np.clip(sB, a_min=jitter)
    Z1 = B / sA.reshape((-1, 1))
    KL1 = np.trace(Z1) / d - 1
    KL2 = (np.sum(np.log(sA)) - np.sum(np.log(sB))) / d
    return np.clip(0.5 * (KL1 + KL2), a_min=0.)

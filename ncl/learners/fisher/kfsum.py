from ncl.learners.fisher.approx import full
import numpy as onp
import jax.numpy as np
from .. import tools
from jax import random, lax, jit


def nearest_kf(B, C):
    """Here we assume that all these matrices are symmetric
    NOT CHECKED explicitly
    """

    BR, BL = B['R'], B['L']
    CR, CL = C['R'], C['L']

    nl = BL.shape[0]
    nr = BR.shape[0]
    vBR = BR.ravel()
    vCR = CR.ravel()
    vBL = BL.ravel()
    vCL = CL.ravel()
    Z = np.outer(vBL, vBR) + np.outer(vCL, vCR)
    Z = Z.reshape((nl * nl, nr * nr))
    u, s, v = np.linalg.svd(Z)
    u = u[:, 0].reshape(BL.shape)
    u = u * np.sign(np.trace(u))
    v = v[0, :].reshape(BR.shape)
    v = v * np.sign(np.trace(v))
    s = s[0]
    L, R = np.sqrt(s) * u, np.sqrt(s) * v
    L, R = 0.5 * (L + L.T), 0.5 * (R + R.T)
    return {'L': L, 'R': R}


@jit
def fast_nearest_kf(B, C):
    """Here we assume that all these matrices are symmetric
    NOT CHECKED explicitly
    """

    BR, BL = B['R'], B['L']
    CR, CL = C['R'], C['L']

    vBR = BR.ravel()
    vCR = CR.ravel()
    vBL = BL.ravel()
    vCL = CL.ravel()
    q, _ = np.linalg.qr(np.vstack((vBL, vCL)).T)
    h = np.outer(q.T @ vBL, vBR) + np.outer(q.T @ vCL, vCR)
    q2, r2 = np.linalg.qr(h.T)
    u, s, vt = np.linalg.svd(r2.T, full_matrices=False)
    r = np.sqrt(s)[0] * (vt[0:1, :] @ q2.T)
    l = np.sqrt(s)[0] * (q @ u)[:, 0]
    R = np.reshape(r, BR.shape)
    R = R * np.sign(np.trace(R))
    L = np.reshape(l, BL.shape)
    L = L * np.sign(np.trace(L))
    return {'L': L, 'R': R}


def additive_nearest_kf(B, C):
    """Here we assume that all these matrices are symmetric
    NOT CHECKED explicitly
    """
    BR, BL = B['R'], B['L']
    CR, CL = C['R'], C['L']

    pi = np.sqrt(np.trace(BL) * np.trace(CR)) / np.sqrt(
        np.trace(CL) * np.trace(BR))
    return {'L': BL + CL * pi, 'R': BR + CR / pi}


def randomized_nearest_kf(key, B, C, k=5):
    """Here we assume that all these matrices are symmetric
    NOT CHECKED explicitly
    """
    BR, BL = B['R'], B['L']
    CR, CL = C['R'], C['L']

    vBR = BR.ravel()[None, :]
    vCR = CR.ravel()[None, :]
    vBL = BL.ravel()[:, None]
    vCL = CL.ravel()[:, None]
    n = vBL.size
    eta = random.normal(key, (k, n))
    Z = ((eta @ vBL) @ vBR) + ((eta @ vCL) @ vCR)  # k x n
    q, _ = np.linalg.qr(Z.T)  # n x k
    Y = (vBL @ (vBR @ q)) + (vCL @ (vCR @ q))  # n x k
    u, s, v = np.linalg.svd(Y, full_matrices=False)
    v = v @ q.T
    u = u[:, 0].reshape(BL.shape)
    u = u * np.sign(np.trace(u))
    v = v[0, :].reshape(BR.shape)
    v = v * np.sign(np.trace(v))
    s = s[0]

    L, R = np.sqrt(s) * u, np.sqrt(s) * v
    L, R = 0.5 * (L + L.T), 0.5 * (R + R.T)
    return {'L': L, 'R': R}


def kl(E, F, BL, BR, CL, CR):
    """KL between
    p1 = N(0, E kron F)
    p2 = N(0, BL kron BR + CL kron CR)
    """
    n1 = BL.shape[0]
    n2 = BR.shape[0]
    kl1 = (n2 * tools.logdet(E)) + (n1 * tools.logdet(F))
    Einv = tools.safe_inv(E)
    Finv = tools.safe_inv(F)
    kl2 = np.trace(BL @ Einv) * np.trace(BR @ Finv)
    kl3 = np.trace(CL @ Einv) * np.trace(CR @ Finv)
    big = np.kron(BL, BR) + np.kron(CL, CR)
    kl4 = -tools.logdet(big)
    return 0.5 * (kl1 + kl2 + kl3 + kl4 - (n1 * n2))


def balance(X):
    XL = X['L']
    XR = X['R']
    scale = (np.trace(XL) * np.trace(XR)) / (XR.shape[0] * XL.shape[0])
    sX = np.sqrt(scale)
    XL = sX * XL / (np.trace(XL) / XL.shape[0])
    XR = sX * XR / (np.trace(XR) / XR.shape[0])
    return {'L': XL, 'R': XR}


def kf_step(E, F, BL, BR, CL, CR, beta=0.5):
    n1 = BL.shape[0]
    n2 = BR.shape[0]
    E = tools.recondition(E)
    F = tools.recondition(F)
    Einv = tools.safe_inv(E)
    Finv = tools.safe_inv(F)
    e1 = np.trace(BR @ Finv) / n2
    e2 = np.trace(CR @ Finv) / n2
    f1 = np.trace(BL @ Einv) / n1
    f2 = np.trace(CL @ Einv) / n1
    dF = f1 * BR + f2 * CR - F
    dE = e1 * BL + e2 * CL - E
    E = E + beta * dE
    F = F + beta * dF
    return E, F, np.mean(dE**2) / np.mean(E**2), np.mean(dF**2) / np.mean(
        F**2), e1, e2, f1, f2


def opt_nearest_kf(key, max_steps, B, C, beta=0.3, verbose=False):
    """Returns optimal KF with respect to KL
    """
    B = balance(B)
    C = balance(C)
    BR, BL = B['R'], B['L']
    BR = tools.recondition(BR)
    BL = tools.recondition(BL)
    CR, CL = C['R'], C['L']
    CR = tools.recondition(CR)
    CL = tools.recondition(CL)

    EF = additive_nearest_kf(B, C)  #initialize from additive
    E = EF['L']
    F = EF['R']

    # First loop
    for i in range(max_steps):
        E, F, dE, dF, e1, e2, f1, f2 = kf_step(E, F, BL, BR, CL, CR, beta=beta)
        if verbose:
            loss = kl(E, F, BL, BR, CL, CR)
            if i % 5 == 0:
                print(i, loss, "e1", e1, "e2", e2, "f1", f1, "f2", f2, dE, dF)
        if (dE < 1e-7 and dF < 1e-7):
            return {'L': E, 'R': F}

    if (dE < 5e-4 and dF < 5e-4):
        return {'L': E, 'R': F}

    # save for debugging
    onp.savetxt("BL", BL)
    onp.savetxt("BR", BR)
    onp.savetxt("CL", CL)
    onp.savetxt("CR", CR)
    raise Exception(f'max iterations reached: dE {dE}, dF {dF}')


def norm_kf(B, C):
    """
    compute (A3 kron B3) \approx (A1 kron B1) + (A2 kron B2)
    A1: prior[r], B1: prior[l]
    A2: fisher[r], B2: fisher[l]
    A3: new_p[r], B3: new_p[l]
    """
    BR, BL = B['R'], B['L']
    CR, CL = C['R'], C['L']

    #minimize upper bound on trace norm
    pi = np.sqrt((np.trace(BL) * np.trace(CR)) / (np.trace(CL) * np.trace(BR)))

    E = BL + pi * CL
    F = BR + (1 / pi) * CR

    return {'L': E, 'R': F}
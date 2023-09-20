import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lalg

# the all-at once system is formed in a sparse matrix M for approach C and solved with gmres directly
def paradiag(Nt, dt, A, M, rr, alpha, tol_paradiag, tol_gmres, maxiter):

    k_paradiag = 0
    dimA = A.shape[0]
    dimM = 2 * Nt * dimA

    # preconditioner
    E = np.eye(Nt, k=-1)
    E[0, -1] = alpha

    E11 = np.zeros((2, 2))
    E11[0, 0] = 1

    E22 = np.zeros((2, 2))
    E22[1, 1] = 1

    P = sp.kron(E11, sp.kron(sp.eye(Nt), sp.eye(dimA) - dt * A) - sp.kron(E, sp.eye(dimA)))
    P += sp.kron(E22, sp.kron(sp.eye(Nt), sp.eye(dimA) - dt * A.transpose()) - sp.kron(E.transpose(), sp.eye(dimA)))

    print('    paradiag its:')
    yp = np.zeros(dimM)
    res = M @ yp - rr
    res_norm = np.linalg.norm(res, np.inf)
    print('    ', k_paradiag, res_norm)
    
    while k_paradiag < maxiter:
        b = (P - M) @ yp + rr
        yp, info = lalg.gmres(P, b, tol=tol_gmres, atol=0)
        res = M @ yp - rr
        res_norm = np.linalg.norm(res, np.inf)
        k_paradiag += 1
        print('    ', k_paradiag, res_norm)

        if res_norm < tol_paradiag:
            break

    return yp, k_paradiag


def paradiag_factorization(Nt, dt, A, M, rr, alpha, tol_paradiag, tol_gmres, maxiter):
    k_paradiag = 0
    dimA = A.shape[0]
    dimM = 2 * Nt * dimA

    # preconditioner
    E = np.eye(Nt, k=-1)
    E[0, -1] = alpha

    E11 = np.zeros((2, 2))
    E11[0, 0] = 1

    E22 = np.zeros((2, 2))
    E22[1, 1] = 1

    P = sp.kron(E11, sp.kron(sp.eye(Nt), sp.eye(dimA) - dt * A) - sp.kron(E, sp.eye(dimA)))
    P += sp.kron(E22, sp.kron(sp.eye(Nt), sp.eye(dimA) - dt * A.transpose()) - sp.kron(E.transpose(), sp.eye(dimA)))

    print('    paradiag its:')
    yp = np.zeros(dimM)
    res = rr - M @ yp
    res_norm = np.linalg.norm(res, np.inf)
    print('    ', k_paradiag, res_norm)

    while k_paradiag < maxiter:
        b = (P - M) @ yp + rr

        # do fft
        g = fft(res, alpha, Nt) # checked until here

        yp, info = lalg.gmres(P, b, tol=tol_gmres, atol=0)
        res = rr - M @ yp
        res_norm = np.linalg.norm(res, np.inf)
        k_paradiag += 1
        print('    ', k_paradiag, res_norm)

        if res_norm < tol_paradiag:
            break

    return yp, k_paradiag


def fft(x, alpha, L):

    Nx = x.shape[0] // (2 * L)
    r1 = x[:x.shape[0] // 2].astype(complex)
    r2 = x[x.shape[0] // 2:].astype(complex)

    # for state scale with J^(-1)
    # for adjoint scale with J
    for l in range(L):
        r1[l * Nx:(l + 1) * Nx] *= alpha ** (l / L)
        r2[l * Nx:(l + 1) * Nx] *= alpha ** (-l / L)

    # do fft
    rr1 = np.zeros_like(r1)
    rr2 = np.zeros_like(r2)
    w = np.exp(-2 * np.pi * 1j / L)
    for j in range(L):
        for k in range(L):
            rr1[j * Nx: (j + 1) * Nx] += w ** (j * k) * r1[k * Nx: (k + 1) * Nx]
            rr2[j * Nx: (j + 1) * Nx] += w ** (j * k) * r2[k * Nx: (k + 1) * Nx]

    y = np.empty_like(x).astype(complex)
    y[:x.shape[0] // 2] = rr1
    y[x.shape[0] // 2:] = rr2

    return y

# TODO write paradiag with a factorization

# TODO write a nonlinear version

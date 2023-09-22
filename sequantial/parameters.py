import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lalg

# globals
Nt = 4
Nx = 4
alpha = 1e-1

# maxiters
max_outer_its = 1
max_paradiag_its = 2

# tols
tol_outer = 5e-4
tol_paradiag = 1e-5
tol_gmres = 1e-6

T = 1e-1 * Nt    # time domain [0,T]
xl = 0  # spatial domain [xl,xr]
xr = 4
gamma = 0.05    # beta from the Guettel/Pearson paper

# discretization
t = np.linspace(0, T, Nt + 1)
x1 = np.linspace(xl, xr, Nx + 1)[:-1]
x2 = np.linspace(xl, xr, Nx + 1)[:-1]
x = np.meshgrid(x1, x2)
dx = (xr-xl) / Nx
dt = T / Nt

# spatial matrix
# A = 2D Laplacian PBC
dimA = Nx
A = 1 / dx ** 2 * (np.diag(-2 * np.ones(dimA), k=0) + np.diag(np.ones(dimA - 1), k=1) + np.diag(np.ones(dimA - 1), k=-1))
A[-1, 0] = 1 / dx ** 2
A[0, -1] = 1 / dx ** 2
A = sp.csr_matrix(A)
A = sp.kron(A, sp.eye(dimA)) + sp.kron(sp.eye(dimA), A)
dimA = A.shape[0]


# test case: Guettel/Pearson 2018, Sect. 6.1
def yd(t, x):
    return ((2 * np.pi ** 2 / 4 + 2 / np.pi ** 2 / gamma) * np.exp(T) + (1 - np.pi ** 2 / 2 - 4 / (4 + 2 * np.pi ** 2) / gamma) * np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)


def y(t, x):
    return (2 / (np.pi ** 2 * gamma) * np.exp(T) - 4 / (4 + 2 * np.pi ** 2) / gamma * np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)


def p(t, x):
    return (np.exp(T) - np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)


def uex(t, x):
    return 1 / gamma * p(t, x)


def grad_equation(u, yp, grad, pT):
    dimA = yp.shape[0] // (2 * Nt)
    dimM = yp.shape[0]
    grad[:-dimA] = gamma * u[:-dimA] - yp[dimM // 2 + dimA:].real
    grad[-dimA:] = gamma * u[-dimA:] - pT
    return grad


def coarse_solve_for_e(Nt, nx):

    dima = nx
    dx = (xr - xl) / nx
    a = 1 / dx ** 2 * (np.diag(-2 * np.ones(dima), k=0) + np.diag(np.ones(dima - 1), k=1) + np.diag(np.ones(dima - 1), k=-1))
    a[-1, 0] = 1 / dx ** 2
    a[0, -1] = 1 / dx ** 2
    a = sp.csr_matrix(a)
    a = sp.kron(a, sp.eye(dima)) + sp.kron(sp.eye(dima), a)
    dima = a.shape[0]

    e = np.zeros(dima)
    Cstar = sp.eye(dima) - dt * a.transpose()
    norm_e_max = 0
    r = dt * np.ones(dima)
    for i in range(Nt-1, -1, -1):
        r += e
        e, info = lalg.gmres(Cstar, dt * np.ones(dima) + e, tol=1e-5, atol=0)
        norm_e_max = max(norm_e_max, np.linalg.norm(e, np.inf))

    return norm_e_max

def evaluate_obj(y, u, yd_vec):
    return (dt * dx ** 2) / 2 * (np.linalg.norm(y - yd_vec, 2) ** 2 + gamma * np.linalg.norm(u, 2) ** 2)

E = coarse_solve_for_e(Nt, 20)
print('E = ', E)

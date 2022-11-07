import numpy as np
np.set_printoptions(linewidth=np.inf)
import scipy as sp
import matplotlib.pyplot as plt
from seq_time_stepping import Newton, IMEX, Parallel_IMEX_refinement
import seaborn as sns
from time import time
import matplotlib.cm as cm
import seaborn as sns

# ALLEN CAHN
#  GLOBAL VARS
EPS = 0.01
R = 0.25
T1 = 0
steps = 8
T2 =  0.0075 / 128 * steps
dt = (T2 - T1) / steps
X1 = -0.5
X2 = 0.5
coll_points = 2
spatial_points = 100

# tolerances
tol = 1e-5
stol = 1e-7
maxiter = 10

# grid and matrix
x1 = np.linspace(X1, X2, spatial_points + 1)[:-1]
x2 = np.linspace(X1, X2, spatial_points + 1)[:-1]
x = np.meshgrid(x1, x2)
dx = (X2 - X1) / spatial_points

print('dt < EPS ^ 2?', dt < EPS ** 2)
print('dt = ', dt, ', EPS = ', EPS)
print('spatial points = ', spatial_points, ', spatial order = ', dx ** 2, ', dt^(2M-1) = ', dt ** (2 * coll_points - 1))

# initial guess
u0 = np.tanh((R - np.sqrt(x[0] ** 2 + x[1] ** 2)) / (np.sqrt(2) * EPS)).flatten()

# Laplacian 2D and 2nd order
data = 1 / dx ** 2 * np.array([np.ones(spatial_points), -2 * np.ones(spatial_points), np.ones(spatial_points)])
A = sp.sparse.spdiags(data, diags=[-1, 0, 1], m=spatial_points, n=spatial_points)
A = sp.sparse.lil_matrix(A)
A[0, -1] = 1 / dx ** 2
A[-1, 0] = 1 / dx ** 2
A = sp.sparse.kron(A, sp.sparse.eye(spatial_points)) + sp.sparse.kron(sp.sparse.eye(spatial_points), A)


# FUNCTIONS
def F(u):
    return 1 / EPS ** 2 * u * (1 - u ** 2)

def dF(u):
    return 1 / EPS ** 2 * (1 - 3 * u ** 2)

def f(u):
    return A @ u + F(u)

def df(u):
    data = dF(u)
    return A + sp.sparse.spdiags(data, diags=0, m=A.shape[0], n=A.shape[1])

# rhs vector
def b(t):
    return np.zeros(spatial_points ** 2)


# seq. IMEX
print('\nIMEX')
print('====')
t_start = time()
u_imex, res_imex, its_imex = IMEX(T1, u0, dt, F, A, b, steps, restol=tol, stol=stol, coll_points=coll_points, maxiter=maxiter)
print('time = ', time() - t_start)
print('maximum residual = ', max(res_imex))
print('iterations = ', its_imex, ', total = ', sum(its_imex))


# seq. Newton
print('\nNewton')
print('========')
t_strat = time()
u_newton, res_newton, its_newton = Newton(T1, u0, dt, f, df, b, steps, restol=tol, stol=stol, coll_points=coll_points, maxiter=maxiter)
print('time = ', time() - t_start)
print('maximum residual = ', max(res_newton))
print('iterations = ', its_newton, ', total = ', sum(its_newton))

print('u_n - u_i', np.linalg.norm(u_imex - u_newton, np.inf))

print('\nParallel_IMEX_refinement')
print('=========')
t_start = time()
u_pimex, u_history_pimex, err_history_pimex, res_history_pimex, cerr_pimex = Parallel_IMEX_refinement(T1, u0, dt, F, dF, A, b, steps, alpha=[1e-8], beta=[1], maxiter=maxiter, coll_points=coll_points, restol=tol, stol=stol, reff_run=u_imex)
print('time = ', time() - t_start)
print('residual = ', res_history_pimex)

#for i in range(spatial_points ** 2):
#    print(i, u_newton[i])











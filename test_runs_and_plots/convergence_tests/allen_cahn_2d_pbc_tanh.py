import numpy as np
np.set_printoptions(linewidth=np.inf)
import scipy as sp
import matplotlib.pyplot as plt
from seq_time_stepping import Newton, IMEX
import seaborn as sns
from time import time
import matplotlib.cm as cm
import seaborn as sns

# ALLEN CAHN
#  GLOBAL VARS
EPS = 0.05
R = 0.25
T1 = 0
steps = 10
dt = EPS ** 2 * 0.1
T2 = steps * dt
X1 = -0.5
X2 = 0.5
coll_points = 1
spatial_points = 80#int(1.5 / ( 4/5 * EPS ))

# tolerances
tol = 1e-5
stol = 1e-6
maxiter = 150

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
    return 1 / EPS ** 2 * u * (1 - u)

def f(u):
    return A @ u + F(u)

def df(u):
    data = 1 / EPS ** 2 * (1 - 2 * u)
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











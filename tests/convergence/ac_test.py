import numpy as np
np.set_printoptions(linewidth=np.inf)
import scipy as sp
import matplotlib.pyplot as plt
from seq_time_stepping import Newton, IMEX, ParalpHa, Parallel_IMEX_refinement
import seaborn as sns
from time import time
import matplotlib.cm as cm
import seaborn as sns

# ALLEN CAHN

# (eps, L) = (0.5, 8) or (0.1, 32) -> work with sol_tol = 1e-12
# (0.1, 64) has weird ptol, seems ok with 0.5 * dt
# or (0.04, 4)

#  GLOBAL VARS
EPS = 0.01
R = 0.25
T1 = 0
steps = 8
T2 = 0.0075 / 128 * steps
dt = (T2 - T1) / steps
X1 = -0.5
X2 = 0.5
coll_points = 2
spatial_points = 100

# tolerances
RESTOL = dict()

tmp = 1e-5
RESTOL['referent'] = tmp
RESTOL['seq imex'] = tmp
RESTOL['seq newton'] = tmp

tmp = 1e-7
SOLTOL = dict()
SOLTOL['referent'] = tmp
SOLTOL['seq imex'] = tmp
SOLTOL['seq newton'] = tmp
SOLTOL['paralpha'] = tmp
SOLTOL['refinement'] = tmp

# maxiters
SEQ_maxiter = 30
PAR_maxiter = 5

# grid and matrix
x1 = np.linspace(X1, X2, spatial_points + 1)[:-1]
x2 = np.linspace(X1, X2, spatial_points + 1)[:-1]
x = np.meshgrid(x1, x2)
dx = (X2 - X1) / spatial_points

print('dt < EPS ^ 2?', dt < EPS ** 2)
print('dt = ', dt, ', EPS = ', EPS)
print('spatial points = ', spatial_points, ', spatial order = ', dx ** 2, ', dt^(2M-1) = ', dt ** (2 * coll_points - 1))

# initial guess
u0 = np.tanh((R - np.sqrt(x[0]**2 + x[1]**2)) / (np.sqrt(2) * EPS)).flatten()

# Laplacian 2D and 2nd order
data = 1 / dx ** 2 * np.array([np.ones(spatial_points), -2 * np.ones(spatial_points), np.ones(spatial_points)])
A = sp.sparse.spdiags(data, diags=[-1, 0, 1], m=spatial_points, n=spatial_points)
A = sp.sparse.csr_matrix(A)
A[0, -1] = 1 / dx ** 2
A[-1, 0] = 1 / dx ** 2
A = sp.sparse.kron(A, sp.sparse.eye(spatial_points)) + sp.sparse.kron(sp.sparse.eye(spatial_points), A)

# FUNCTIONS

def f(u):
    return A @ u + F(U)

def df(u):
    data = dF(u)
    return A + sp.sparse.spdiags(data, diags=0, m=A.shape[0], n=A.shape[1])

def F(u):
    return 1 / EPS ** 2 * u * (1 - u**2)

def dF(u):
    data = 1 / EPS ** 2 * (1 - 3 * u ** 2)
    return sp.sparse.spdiags(data, diags=0, m=data.shape[0], n=data.shape[0])

# rhs vector
def b(t):
    return np.zeros(spatial_points ** 2)


sol = dict()
sol['referent'] = dict()
sol['seq imex'] = dict()
sol['seq newton'] = dict()
sol['paralpha'] = dict()
sol['refinement'] = dict()

# reference sol
print('\nreference solution')
print('==========================')
#sol['referent']['u'], sol['referent']['residuals'], sol['referent']['ncount'] = Newton(T1, u0, dt / 10, f, df, b, 10 * steps, restol=RESTOL['referent'], stol=SOLTOL['referent'], coll_points=coll_points, maxiter=SEQ_maxiter)
sol['referent']['u'], sol['referent']['residuals'], sol['referent']['ncount'] = IMEX(T1, u0, dt / 10, F, A, b, 10 * steps, restol=RESTOL['referent'], stol=SOLTOL['referent'], coll_points=coll_points, maxiter=SEQ_maxiter)
print('maximum residual = ', max(sol['referent']['residuals']))
print('iterations = ', sol['referent']['ncount'], ', total = ', sum(sol['referent']['ncount']))

# seq. IMEX
print('\nIMEX')
print('====')
t_start = time()
sol['seq imex']['u'], sol['seq imex']['residuals'], sol['seq imex']['ncount'] = IMEX(T1, u0, dt, F, A, b, steps, restol=RESTOL['seq imex'], stol=SOLTOL['seq imex'], coll_points=coll_points, maxiter=SEQ_maxiter)
print('time = ', time() - t_start)
print('maximum residual = ', max(sol['seq imex']['residuals']))
print('iterations = ', sol['seq imex']['ncount'], ', total = ', sum(sol['seq imex']['ncount']))


# seq. Newton
'''
print('\nNewton')
print('========')
t_strat = time()
sol['seq newton']['u'], sol['seq newton']['residuals'], sol['seq newton']['ncount'] = Newton(T1, u0, dt, f, df, b, steps, restol=RESTOL['seq newton'], stol=SOLTOL['seq newton'], coll_points=coll_points, maxiter=SEQ_maxiter)
print('time = ', time() - t_start)
print('maximum residual = ', max(sol['seq newton']['residuals']))
print('iterations = ', sol['seq newton']['ncount'], ', total = ', sum(sol['seq newton']['ncount']))
'''

pTOL = 3 / 2 * np.linalg.norm(sol['referent']['u'] - sol['seq imex']['u'], np.inf)
print('\npTOL = ', pTOL)

m0 = steps * dt
optimal = False
alphas = [1e-12]# * 5 +[1e-3] * 3 + [1e-2] + [1e-1]
beta = [1]

print('\nParalpHa')
print('=========')
t_start = time()
sol['paralpha']['u'], sol['paralpha']['residual'], sol['paralpha']['ncount'], sol['paralpha']['u history'], sol['paralpha']['residual history'], sol['paralpha']['c'], sol['paralpha']['Lip'], sol['paralpha']['alpha'], sol['paralpha']['m history'], sol['paralpha']['m history corr'], sol['paralpha']['consecutive errors'] = ParalpHa(T1, u0, dt, F, dF, A, b, steps, alpha=alphas, beta=beta, maxiter=PAR_maxiter, coll_points=coll_points, restol=pTOL, stol=SOLTOL['paralpha'], reff_run=sol['referent']['u'], const=1, m0=m0, optimal=optimal)
print('time = ', time() - t_start)
print('maximum residual = ', sol['paralpha']['residual'])
print('iterations = ', sol['paralpha']['ncount'], ', total = ', sol['paralpha']['ncount'])
print('err = ', np.linalg.norm(sol['referent']['u'] - sol['paralpha']['u'], np.inf))

print('\nParallel_IMEX_refinement')
print('=========')
t_start = time()
sol['refinement']['u'], sol['refinement']['u history'], sol['refinement']['error history'], sol['refinement']['residual history'], sol['refinement']['consecutive errors'] = Parallel_IMEX_refinement(T1, u0, dt, F, dF, A, b, steps, alpha=alphas, beta=beta, maxiter=PAR_maxiter, coll_points=coll_points, restol=pTOL, stol=SOLTOL['paralpha'], reff_run=sol['referent']['u'], const=1, m0=m0, optimal=optimal)
print('time = ', time() - t_start)
print('residual = ', sol['paralpha']['residual history'])
print('iterations = ', sol['paralpha']['ncount'], ', total = ', sol['paralpha']['ncount'])
print('err = ', np.linalg.norm(sol['referent']['u'] - sol['refinement']['u'], np.inf))

plt.semilogy(sol['paralpha']['residual history'], 'x-')
plt.semilogy(sol['refinement']['error history'], 'x:')
plt.semilogy(np.ones(PAR_maxiter) * pTOL, color='gray', linestyle='--')
plt.legend(['pimex', 'refined pimex'])
plt.show()
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lalg
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf, precision=1)
print('I started...')

# globals
T = 1
Nt = 10
Nx = 50
alpha = 1e-3
gamma = 0.04

# maxiters
max_outer_its = 1000
max_paralpha_its = 10

# tols
tol_outer = 1e-14
tol_paralpha = 1e-6
tol_gmres = 1e-10


def yd(t, x):
    return ((np.pi**2 / 4 + 4 / np.pi**2 / gamma) * np.exp(T) + (1 - np.pi**2 / 4 - 4 / (4 + np.pi**2) / gamma) * np.exp(t)) * np.cos(np.pi * x / 2)


def y(t, x):
    return (4 / (np.pi**2 * gamma) * np.exp(T) - 4 / (4 + np.pi**2) / gamma * np.exp(t)) * np.cos(np.pi * x / 2)


dimA = Nx
dimM = 2 * dimA * Nt

t = np.linspace(0, T, Nt + 1)
x = np.linspace(0, 4, Nx + 1)[:-1]
dx = 4 / Nx
dt = 1 / Nt

# matrices
# A = 1D Laplacian PBC
A = 1 / dx**2 * (np.diag(-2 * np.ones(dimA), k=0) + np.diag(np.ones(dimA - 1), k=1) + np.diag(np.ones(dimA - 1), k=-1))
A[-1, 0] = 1 / dx**2
A[0, -1] = 1 / dx**2

# M = all at once coupled system
M = np.zeros((dimM, dimM))
M[:dimM // 2, :dimM // 2] = np.kron(np.eye(Nt), np.eye(dimA) - dt * A) - np.kron(np.eye(Nt, k=-1), np.eye(dimA))
M[dimM // 2:, dimM // 2:] = np.kron(np.eye(Nt), np.eye(dimA) - dt * A.transpose()) - np.kron(np.eye(Nt, k=1), np.eye(dimA))
M[dimM // 2:, :dimM // 2] = dt * np.kron(np.fliplr(np.eye(Nt, k=1)), np.eye(dimA))

# P = block alpha Jacobi
P = np.zeros_like(M)
P[:dimM // 2, :dimM // 2] = M[:dimM // 2, :dimM // 2]
P[dimM // 2:, dimM // 2:] = M[dimM // 2:, dimM // 2:]
P[:dimA, dimM // 2 - dimA:dimM // 2] = -alpha * np.eye(dimA)
P[dimM // 2:dimM // 2 + dimA, -dimA:] = -alpha * np.eye(dimA)

P = sp.csr_matrix(P)
M = sp.csr_matrix(M)

# initial conditions
y0 = y(0, x)#(4 / (np.pi**2 * gamma) * np.exp(T) - 4 / (4 + np.pi**2) / gamma) * np.cos(np.pi * x / 2)
p0 = np.zeros(dimA)

# C iterations
u = np.zeros(dimM // 2)
r = np.zeros(dimM)
yp = np.zeros(dimM)

# initial guess
yp[:dimM // 2] = np.tile(y0, Nt)
yp[dimM // 2:] = np.tile(p0, Nt)

r[:dimA] += y0
r[dimM // 2:dimM // 2 + dimA] += p0
r[-dimA:] -= dt * y0

for i in range(Nt):
    r[dimM // 2 + i * dimA:dimM // 2 + (i + 1) * dimA] += dt * yd(t[Nt - i - 1], x)

grad_norms_history = []

# C iterations
k_outer_its = 0
while k_outer_its < max_outer_its:
    rr = r.copy()
    rr[:dimM // 2] += dt * u

    # paralpha iterations
    '''
    k_paralpha = 0
    while k_paralpha < max_paralpha_its:
        # solve big system with gmres (without factoriazation)
        yp, info = lalg.gmres(P, (P - M) @ yp + rr, tol=tol_gmres)
        res = M @ yp - rr
        res_norm = np.linalg.norm(res, np.inf)
        print(k_outer_its, k_paralpha, res_norm)

        if res_norm < tol_paralpha:
            break

        k_paralpha += 1
    '''
    yp, info = lalg.gmres(M, rr, tol=tol_gmres)

    grad = gamma * u + yp[dimM // 2:]
    grad_norm_scaled = 1 / (dt * dx) * np.linalg.norm(grad, 2)
    grad_norms_history.append(grad_norm_scaled)
    print(k_outer_its, grad_norm_scaled)

    if grad_norm_scaled <= tol_outer:
        break

    u = u - grad
    k_outer_its += 1

exact_sol = np.zeros(dimM // 2)
for i in range(Nt):
    exact_sol[i * dimA:(i + 1) * dimA] = y(t[i + 1], x)

error_to_solution = np.linalg.norm(yp[:dimM // 2] - exact_sol, np.inf)
print('error = ', error_to_solution)

plt.plot(yp[:dimM // 2])
plt.plot(exact_sol)
plt.legend(['approx', 'exact'])
#plt.semilogy(grad_norms_history)
plt.show()






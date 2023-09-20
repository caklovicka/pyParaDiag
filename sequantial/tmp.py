import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from some_functions import C_paradiag, get_M

np.set_printoptions(linewidth=np.inf, precision=2, threshold=sys.maxsize)
print('I started...')

# globals
Nt = 40
Nx = 40
alpha = 1e-3

# maxiters
max_outer_its = 30
max_paradiag_its = 10

# tols
tol_outer = 1e-5
tol_paradiag = 1e-7
tol_gmres = 1e-8

# test case: Guettel/Pearson 2018, Sect. 6.1

T = 1    # time domain [0,T]
xl = -2  # spatial domain [xl, xr]
xr = 2
gamma = 0.05 # beta from the Guettel/Pearson paper


def yd(t, x):
    return ((2 * np.pi ** 2 / 4 + 2 / np.pi ** 2 / gamma) * np.exp(T) + (1 - np.pi ** 2 / 2 - 4 / (4 + 2 * np.pi ** 2) / gamma) * np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)


def y(t, x):
    return (2 / (np.pi ** 2 * gamma) * np.exp(T) - 4 / (4 + 2 * np.pi ** 2) / gamma * np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)


def p(t, x):
    return (np.exp(T) - np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)


def grad_equation(u, yp):
    grad[:-dimA] = gamma * u[:-dimA] - yp[dimM // 2 + dimA:]
    grad[-dimA:] = gamma * u[-dimA:] - pT
    return grad


def uex(t, x):
    return 1 / gamma * p(t, x)


t = np.linspace(0, T, Nt + 1)
x1 = np.linspace(xl, xr, Nx + 1)[:-1]
x2 = np.linspace(xl, xr, Nx + 1)[:-1]
x = np.meshgrid(x1, x2)
dx = (xr-xl) / Nx
dt = T / Nt

# matrices
# A = 2D Laplacian PBC
dimA = Nx
A = 1 / dx ** 2 * (np.diag(-2 * np.ones(dimA), k=0) + np.diag(np.ones(dimA - 1), k=1) + np.diag(np.ones(dimA - 1), k=-1))
A[-1, 0] = 1 / dx ** 2
A[0, -1] = 1 / dx ** 2
A = sp.csr_matrix(A)
A = sp.kron(A, sp.eye(Nx)) + sp.kron(sp.eye(Nx), A)

dimA = Nx ** 2
dimM = 2 * dimA * Nt

# initial conditions
y0 = y(0, x).flatten()
pT = p(T, x).flatten()

# zz = y(0,x)
# print(np.shape(zz))
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x[0], x[1], zz, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# plt.show()

# auxilaries
u = np.zeros(dimM // 2)
r = np.zeros(dimM)
yp = np.zeros(dimM)
grad = np.zeros(dimM // 2)

# exact solutions
exact_y = np.zeros(dimM // 2)
for i in range(Nt):
    exact_y[i * dimA:(i + 1) * dimA] = y(t[i + 1], x).flatten()

exact_p = np.zeros(dimM // 2)
for i in range(Nt):
    exact_p[i * dimA:(i + 1) * dimA] = p(t[i], x).flatten()

yd_vec = np.zeros(dimM // 2)
for i in range(Nt):
    yd_vec[i * dimA:(i + 1) * dimA] = yd(t[i + 1], x).flatten()

# initial guesses
yp[:dimM // 2] = np.tile(y0, Nt)
yp[dimM // 2:] = np.tile(pT, Nt)

r[:dimA] += y0
r[-dimA:] += pT
r[dimM // 2:dimM // 2 + dimA] -= dt * y0

for i in range(Nt):
    r[dimM // 2 + i * dimA:dimM // 2 + (i + 1) * dimA] += dt * yd(t[i], x).flatten()

# test: start with exact control, grad norm and errors should be very small
# also: evaluate relative error in control as in Guettel/Pearson
u_exact = np.zeros(dimM // 2)
for i in range(Nt):
    u_exact[i * dimA: (i + 1) * dimA] = uex(t[i + 1], x).flatten()
u = u_exact

M = get_M(dt, Nt, A)
# C iterations
k_outer_its = 0
grad_norms_history = []
obj_history = []
while k_outer_its < max_outer_its:
    rr = r.copy()
    rr[:dimM // 2] += dt * u

    yp = C_paradiag(M, Nt, dt, A, rr, yp, alpha, tol_paradiag, tol_gmres, max_paradiag_its)

    # compute gradient
    grad = grad_equation(u, yp)
    grad_norm_scaled = np.sqrt(dt * dx ** 2) * np.linalg.norm(grad, 2)  # we integrate in space over a 2D domain, so scale by dx1*dx2 = dx**2 (for the squared L2 norm, take sqrt of scaling)
    grad_norms_history.append(grad_norm_scaled)

    # evaluate objective functional
    obj = (dt * dx ** 2) * (np.linalg.norm(yp[:dimM // 2] - yd_vec, 2) ** 2 / 2 + gamma / 2 * np.linalg.norm(u, 2) ** 2)
    obj_history.append(obj)

    error_y = np.linalg.norm(yp[:dimM // 2] - exact_y, np.inf)
    error_p = np.linalg.norm(yp[dimM // 2:] - exact_p, np.inf)
    print(k_outer_its, 'grad =', grad_norm_scaled, ', error_y =', error_y, ', error_p =', error_p, ', objective =', obj)

    if grad_norm_scaled <= tol_outer:
        break

    u = u - grad
    k_outer_its += 1


# relative error in computed control
rel_err_u = np.linalg.norm(u - u_exact, np.inf) / np.linalg.norm(u, np.inf)
print('relative error in computed control', rel_err_u)

# what would be the objective functions value for the exact control?
exact_obj = (dt * dx ** 2) * (np.linalg.norm(yp[:dimM // 2] - yd_vec, 2) ** 2 / 2 + gamma / 2 * np.linalg.norm(u, 2) ** 2)
print('objective for exact control', exact_obj)


# plots
plt.subplot(121)
plt.semilogy(grad_norms_history)
plt.title('norms of gradients')
plt.xlabel('Iteration k')
plt.ylabel("$||j'(u^k)||$")

plt.subplot(122)
plt.semilogy(obj_history)
plt.title('objective function')
plt.xlabel('Iteration k')
plt.ylabel("$J(y(u^k),u^k)$")
plt.show()
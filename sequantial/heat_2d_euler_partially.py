import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from some_functions import paradiag_factorization, paradiag
from parameters import *

np.set_printoptions(linewidth=np.inf, precision=5, threshold=sys.maxsize)
print('I started...')

# M = all at once coupled system
E11 = np.zeros((2, 2))
E11[0, 0] = 1

E21 = np.zeros((2, 2))
E21[1, 0] = 1

E22 = np.zeros((2, 2))
E22[1, 1] = 1

dimA = Nx ** 2
dimM = 2 * dimA * Nt

M = sp.kron(E11, sp.kron(sp.eye(Nt), sp.eye(dimA) - dt * A) - sp.kron(sp.eye(Nt, k=-1), sp.eye(dimA)))
M += sp.kron(E22, sp.kron(np.eye(Nt), sp.eye(dimA) - dt * A.transpose()) - sp.kron(sp.eye(Nt, k=1), sp.eye(dimA)))
M += sp.kron(E21, dt * sp.kron(np.eye(Nt, k=-1), sp.eye(dimA)))


# initial conditions
y0 = y(0, x).flatten()
pT = p(T, x).flatten()

# zz = y(0,x)
# print(np.shape(zz))
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x[0], x[1], zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.show()

# auxilaries
u = np.zeros(dimM // 2)
r = np.zeros(dimM)
yp = np.zeros(dimM).astype(complex)
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

# test: start with exact control, grad norm and errors should be very small
# also: evaluate relative error in control as in Guettel/Pearson
u_exact = np.zeros(dimM // 2)
for i in range(Nt):
    u_exact[i * dimA: (i + 1) * dimA] = uex(t[i + 1], x).flatten()
#u = u_exact

# initial guesses
yp[:dimM // 2] = np.tile(y0, Nt)
yp[dimM // 2:] = np.tile(pT, Nt)

r[:dimA] += y0
r[-dimA:] += pT
r[dimM // 2:dimM // 2 + dimA] -= dt * y0

for i in range(Nt):
    r[dimM // 2 + i * dimA:dimM // 2 + (i + 1) * dimA] += dt * yd(t[i], x).flatten()

# partially coupled iterations
k_outer_its = 0
grad_norms_history = []
obj_history = []

# adaptivity of tolerances
tol_paradiag_adaptive = tol_paradiag
ADAPTIVITY = False
step = 1

total_paradiag_iters = 0

res = r - M @ yp

while k_outer_its < max_outer_its:

    if k_outer_its == 0:
        rr = r.copy()
        rr[:dimM // 2] += dt * u
        
        yp0 = yp.copy()
        yp, k_paradiag = paradiag_factorization(Nt, dt, A, M, rr, alpha, tol_paradiag_adaptive, tol_paradiag_adaptive / 10, max_paradiag_its)
        #yp, k_paradiag = paradiag(Nt, dt, A, M, rr, alpha, tol_paradiag_adaptive, tol_paradiag_adaptive / 10, max_paradiag_its)
        
        total_paradiag_iters += k_paradiag

        # compute gradient
        grad = grad_equation(u, yp, grad, pT)
        grad_norm_scaled = np.sqrt(dt * dx ** 2) * np.linalg.norm(grad, 2)  # we integrate in space over a 2D domain, so scale by dx1*dx2 = dx**2 (for the squared L2 norm, take sqrt of scaling)
        grad_norms_history.append(grad_norm_scaled)

        # evalueate the objective
        obj = evaluate_obj(yp[:dimM // 2], u, yd_vec)
        obj_history.append(obj)

        # errors
        error_y = np.linalg.norm(yp[:dimM // 2] - exact_y, np.inf)
        error_p = np.linalg.norm(yp[dimM // 2:] - exact_p, np.inf)
        print(k_outer_its, 'grad =', grad_norm_scaled, ', error_y =', error_y, ', error_p =', error_p, ', objective =', obj)

        k_outer_its += 1

    if grad_norm_scaled <= tol_outer:
        break

    # update u
    u = u - step * grad

    #u_try = u - step * grad
    #rr = r.copy()
    #rr[:dimM // 2] += dt * u#u_try
    #yp, k_paradiag = paradiag(Nt, dt, A, M, rr, alpha, tol_paradiag_adaptive, tol_paradiag_adaptive / 10, max_paradiag_its)
    #print('         ', k_paradiag)
    #total_paradiag_iters += k_paradiag
    #obj = evaluate_obj(yp, u_try, yd_vec, dimM)
    k_outer_its += 1
'''
    if obj < obj_history[-1] - 1e-3 * step * grad_norm_scaled ** 2:
        obj_history.append(obj)
        u = u_try.copy()
        # compute gradient
        grad = grad_equation(u, yp, grad, pT)
        grad_norm_scaled = np.sqrt(dt * dx ** 2) * np.linalg.norm(grad, 2)  # we integrate in space over a 2D domain, so scale by dx1*dx2 = dx**2 (for the squared L2 norm, take sqrt of scaling)
        grad_norms_history.append(grad_norm_scaled)
        step = 1

        # errors
        error_y = np.linalg.norm(yp[:dimM // 2] - exact_y, np.inf)
        error_p = np.linalg.norm(yp[dimM // 2:] - exact_p, np.inf)
        print(k_outer_its, 'grad =', grad_norm_scaled, ', error_y =', error_y, ', error_p =', error_p, ', objective =', obj)

        k_outer_its += 1

        # set tolerances for the next iter
        if k_outer_its >= 2 and ADAPTIVITY:
            theta = grad_norms_history[-1] ** 2 / grad_norms_history[-2]
            tol_paradiag_adaptive = min(0.1 * theta / (E + 1), 1e-3)
            print('paradiag tolerance for k = {} is {}'.format(k_outer_its, tol_paradiag_adaptive))

    else:
        print('     step not accepted for k = {}, obj = {}, tol = {}, step = {}'.format(k_outer_its, obj, tol_paradiag_adaptive, step))
        tol_paradiag_adaptive /= 10
        step /= 2
 '''

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
plt.plot(obj_history)
plt.title('objective function')
plt.xlabel('Iteration k')
plt.ylabel("$J(y(u^k),u^k)$")
#plt.show()

print('total paradiag its:', total_paradiag_iters)

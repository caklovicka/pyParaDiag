import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from problem_examples_parallel.schrodinger_2d_central2 import Schrodinger
from problem_examples_parallel.advection_2d_pbc_upwind1 import Advection as Advection1
from problem_examples_parallel.advection_2d_pbc_upwind2 import Advection as Advection2
from problem_examples_parallel.advection_2d_pbc_upwind3 import Advection as Advection3
from problem_examples_parallel.heat_2d_pbc_central2 import Heat
from problem_examples_parallel.heat_2d_pbc_central4 import Heat as Heat4
from problem_examples_parallel.heat_2d_pbc_central6 import Heat as Heat6
from problem_examples_parallel.wave_2d_central2 import Wave
from problem_examples_parallel.wave_2d_pbc_central4 import Wave as Wave4
from problem_examples_parallel.schrodinger_2d_central2 import Schrodinger
from problem_examples_parallel.schrodinger_2d_central4 import Schrodinger as Schrodinger4

prob = Advection3()
N = 1000
prob.spatial_points = [N, N]
prob.tol = 1e-5
prob.proc_col = 1
prob.time_intervals = 1
prob.rolling = 1000
prob.proc_row = prob.time_intervals
prob.time_points = 1
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 1e-1
prob.solver = 'custom'
prob.maxiter = 5
prob.smaxiter = 20
prob.stol = 1e-10
prob.m0 = 1 * (prob.T_end - prob.T_start)

prob.setup()
# eq = np.linalg.eigvals(prob.Q)
# ea, v = sp.sparse.linalg.eigs(prob.Apar)
# mini = np.inf
# for i in eq:
#     for j in ea:
#         if mini > np.abs(1 - prob.dt * i * j):
#             mini = np.abs(1 - prob.dt * i * j)
# #
#
print('cfl = ', prob.dt/prob.dx[0])
# print('stability = ', mini)
prob.solve()
# print('cfl = ', prob.dt/prob.dx[0])
# print('stability = ', mini)
# prob.summary(details=True)

exact = prob.u_exact(prob.T_end, prob.x).flatten()
approx = prob.u_last_loc.flatten()
#
# n = 10
# exact_r = exact.real.reshape(prob.spatial_points)
# approx_r = approx.real.reshape(prob.spatial_points)
# col = sns.color_palette("coolwarm", n+5)
# plt.subplot(231)
# plt.contourf(exact_r, levels=n, colors=col)
# plt.colorbar()
# plt.title('exact real')
#
# plt.subplot(232)
# plt.contourf(approx_r, levels=n, colors=col)
# plt.colorbar()
# plt.title('approx real')
#
# plt.subplot(233)
# plt.contourf(exact_r - approx_r, levels=n, colors=col)
# plt.colorbar()
# plt.title('diff real')
#
# exact_r = exact.imag.reshape(prob.spatial_points)
# approx_r = approx.imag.reshape(prob.spatial_points)
# plt.subplot(234)
# plt.contourf(exact_r, levels=n, colors=col)
# plt.colorbar()
# plt.title('exact imag')
#
# plt.subplot(235)
# plt.contourf(approx_r, levels=n, colors=col)
# plt.colorbar()
# plt.title('approx imag')
#
# plt.subplot(236)
# plt.contourf(exact_r - approx_r, levels=n, colors=col)
# plt.colorbar()
# plt.title('diff imag')
#
# plt.show()



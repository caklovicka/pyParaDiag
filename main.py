import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import sys
sys.path.append('../')    # for pySDC on Juwels
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns


from problem_examples_parallel.schrodinger_2d_central2 import Schrodinger
from problem_examples_parallel.advection_2d_pbc_upwind1 import Advection as Advection1
from problem_examples_parallel.advection_2d_pbc_upwind2 import Advection as Advection2
from problem_examples_parallel.advection_2d_pbc_upwind3 import Advection as Advection3
from problem_examples_parallel.advection_2d_pbc_upwind4 import Advection as Advection4
from problem_examples_parallel.advection_2d_pbc_upwind5 import Advection as Advection5
from problem_examples_parallel.heat_2d_pbc_central2 import Heat
from problem_examples_parallel.heat_2d_pbc_central4 import Heat as Heat4
from problem_examples_parallel.heat_2d_pbc_central6 import Heat as Heat6
from problem_examples_parallel.schrodinger_2d_central2 import Schrodinger
from problem_examples_parallel.schrodinger_2d_0_central2 import Schrodinger as Schrodinger0
from problem_examples_parallel.schrodinger_2d_central4 import Schrodinger as Schrodinger4
from problem_examples_parallel.schrodinger_2d_0_central4 import Schrodinger as Schrodinger04
from problem_examples_parallel.schrodinger_2d_0_forward4 import Schrodinger as Schrodinger04_forward
from problem_examples_parallel.schrodinger_2d_0_central6 import Schrodinger as Schrodinger06

np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)

prob = Schrodinger04_forward()
N = 1100
prob.spatial_points = [N, N]
prob.tol = 1e-9
prob.proc_col = 22
prob.time_intervals = 1
prob.rolling = 64
prob.proc_row = 1
prob.time_points = 2
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 0.0032
prob.solver = 'custom'
prob.maxiter = 5
prob.smaxiter = 300
prob.stol = 1e-11
prob.m0 = 10 * (prob.T_end - prob.T_start)

prob.setup()

e = sp.linalg.eigvals(prob.Q)
print(e)
min_eig = np.inf

for ee in e:
    # FORWARD DIFFERENCES
    eig_L = 1/prob.dx[0]** 2 * 15/4 * 2
    # eig_L = 1/prob.dx[0]** 2 * 469/90

    eig = 1 - prob.c * prob.dt * ee * eig_L
    if abs(eig) < min_eig:
        min_eig = abs(eig)

lambd = prob.dt * eig_L * prob.c
print('ro = ', min_eig, 'lambda = ', lambd)

# T = prob.T_end
# res = prob.Apar @ prob.u_exact(T, prob.x).flatten() - 1j * prob.ddu(T, prob.x).flatten()
# err_i = np.linalg.norm(res.imag, np.inf)
# err_r = np.linalg.norm(res.real, np.inf)
# print(err_r, err_i)

prob.solve()
prob.summary(details=True)

#
# plt.subplot(211)
# plt.contourf(prob.u_exact(T, prob.x).flatten().real.reshape(prob.spatial_points), levels=10)
# plt.title('real')
# plt.colorbar()
#
# plt.subplot(212)
# plt.contourf(prob.u_exact(T, prob.x).flatten().imag.reshape(prob.spatial_points), levels=10)
# plt.title('imag')
# plt.colorbar()
#
# plt.show()

# if prob.rank == prob.size - 1:
#     exact = prob.u_exact(prob.T_end, prob.x).flatten()[prob.row_beg:prob.row_end]
#     approx = prob.u_last_loc.flatten()
#     d = exact - approx
#     d = d.flatten()
#     err_abs = np.linalg.norm(d, np.inf)
#     print('abs err  = {}'.format(err_abs))

#
# exact = prob.u_exact(prob.T_end, prob.x)
# approx = prob.u_last_loc.reshape(prob.spatial_points)
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



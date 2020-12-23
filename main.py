import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import sys
import scipy.sparse as sp
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
from problem_examples_parallel.schrodinger_2d_0_central6 import Schrodinger as Schrodinger06

sys.path.append('../')    # for pySDC on Juwels
sys.path.append('/etc/alternatives/petsc4py')

prob = Advection3()
N = 700
prob.spatial_points = [N, N]
prob.tol = 1e-9
prob.proc_col = 1
prob.time_intervals = 1
prob.rolling = 32
prob.proc_row = prob.time_intervals
prob.time_points = 2
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 0.64e-2
prob.solver = 'gmres'
prob.maxiter = 5
prob.smaxiter = 100
prob.stol = 1e-13
prob.m0 = 1 * (prob.T_end - prob.T_start)

prob.setup()
prob.solve()
# prob.summary(details=True)


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



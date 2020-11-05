import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
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
from problem_examples_parallel.wave_2d_central2 import Wave
from problem_examples_parallel.wave_2d_pbc_central4 import Wave as Wave4
from problem_examples_parallel.schrodinger_2d_central2 import Schrodinger
from problem_examples_parallel.schrodinger_2d_0_central2 import Schrodinger as Schrodinger0
from problem_examples_parallel.schrodinger_2d_central4 import Schrodinger as Schrodinger4
from problem_examples_parallel.schrodinger_2d_0_central4 import Schrodinger as Schrodinger04
from problem_examples_parallel.schrodinger_2d_0_central6 import Schrodinger as Schrodinger06

prob = Schrodinger06()
N = 2000
prob.spatial_points = [N, N]
prob.tol = 1e-5
prob.proc_col = 24
prob.time_intervals = 1
prob.rolling = 4
prob.proc_row = prob.time_intervals
prob.time_points = 3
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 5e-3
prob.solver = 'custom'
prob.maxiter = 5
prob.smaxiter = 20
prob.stol = 1e-13
prob.m0 = 1 * (prob.T_end - prob.T_start)

prob.setup()
prob.solve()

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



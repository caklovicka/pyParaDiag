from problem_examples_parallel.schrodinger_2d_central2 import Schrodinger
from problem_examples_parallel.advection_2d_central2 import Advection
from problem_examples_parallel.advection_2d_pbc_central4 import Advection as Advection4
from problem_examples_parallel.heat_2d_pbc_central2 import Heat
from problem_examples_parallel.heat_2d_pbc_central4 import Heat as Heat4
from problem_examples_parallel.wave_2d_central2 import Wave
from problem_examples_parallel.wave_2d_pbc_central4 import Wave as Wave4
from problem_examples_parallel.schrodinger_2d_central2 import Schrodinger
from problem_examples_parallel.schrodinger_2d_central4 import Schrodinger as Schrodinger4
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
prob = Schrodinger4()
N = 634
prob.spatial_points = [N, N]
prob.tol = 1e-12
prob.proc_col = 1
prob.proc_row = 32
prob.time_intervals = 32
prob.rolling = 1
prob.time_points = 1
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 2.4e-07
prob.solver = 'custom'
prob.maxiter = 5
prob.smaxiter = 300
prob.stol = 1e-15
prob.m0 = 10 * (prob.T_end - prob.T_start)

prob.setup()
prob.solve()
prob.summary(details=True)

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



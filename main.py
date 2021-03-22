import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from problem_examples_parallel.advection_2d_pbc_upwind1 import Advection as Advection1
from problem_examples_parallel.advection_2d_pbc_upwind2 import Advection as Advection2
from problem_examples_parallel.advection_2d_pbc_upwind3 import Advection as Advection3
from problem_examples_parallel.advection_2d_pbc_upwind4 import Advection as Advection4
from problem_examples_parallel.advection_2d_pbc_upwind5 import Advection as Advection5
from problem_examples_parallel.heat_2d_pbc_central2 import Heat
from problem_examples_parallel.heat_2d_pbc_central4 import Heat as Heat4
from problem_examples_parallel.heat_2d_pbc_central6 import Heat as Heat6
from problem_examples_parallel.schrodinger_2d_0_forward4 import Schrodinger as Schrodinger04_forward

prob = Advection5()
prob.spatial_points = [700, 700]
prob.tol = 1e-12
prob.proc_col = 1
prob.time_intervals = 1
prob.rolling = 1
prob.proc_row = 64
prob.time_points = 3
prob.optimal_alphas = False
prob.alphas = [0.09131737493714501]
prob.T_start = 0
prob.T_end = 0.0128
prob.solver = 'custom'
prob.maxiter = 6
prob.smaxiter = 50
prob.stol = 1e-16
prob.m0 = 10 * (prob.T_end - prob.T_start)

prob.setup()
prob.solve()
prob.summary(details=True)

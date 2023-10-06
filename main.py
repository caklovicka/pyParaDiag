# the following lines disable the numpy multithreading [optional]
import os
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from examples.heat_2d_pbc_central2 import Heat
prob = Heat()

# choosing a number of points
prob.spatial_points = [40, 40]
prob.collocation_points = 1
prob.T_start = 0
prob.T_end = 1e-1 * 8
prob.time_intervals = 8

#prob.proc_col = 2
prob.proc_row = prob.time_intervals
prob.solver = 'gmres'

prob.outer_maxiter = 3
prob.paradiag_maxiter = 10
prob.solver_maxiter = 50

prob.alpha = 1e-1
prob.paradiag_tol = 1e-5
prob.outer_tol = 5e-4
prob.solver_tol = 1e-6

prob.setup()
prob.solve()
prob.summary(details=False)

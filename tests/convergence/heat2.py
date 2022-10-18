# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../..')

from problem_examples_parallel.heat_2d_pbc_central4 import Heat
prob = Heat()

# choosing a number of points
prob.spatial_points = [400, 400]
prob.time_points = 2

# choosing a time domain
prob.T_start = 0
prob.T_end = 0.1

# choosing the number of intervals handled in parallel
prob.time_intervals = 32
prob.rolling = 1

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'gmres'

# setting maximum number of iterations
prob.maxiter = 8
prob.smaxiter = 50

# choosing a setting for the alpha sequence
prob.optimal_alphas = True
prob.alphas = [1e-12]

# setting tolerances
prob.tol = 1e-9
prob.stol = 1e-10

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)

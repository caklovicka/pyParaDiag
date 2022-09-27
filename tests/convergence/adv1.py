# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../..')

from problem_examples_parallel.advection_2d_pbc_upwind2 import Advection
prob = Advection()

# choosing a number of points
prob.spatial_points = [350, 350]
prob.time_points = 2

# choosing a time domain
prob.T_start = 0
prob.T_end = 1e-2

# choosing the number of intervals handled in parallel
prob.time_intervals = 8
prob.rolling = 1

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'gmres'

# setting maximum number of iterations
prob.maxiter = 10
prob.smaxiter = 50

# choosing a setting for the alpha sequence
prob.optimal_alphas = False
prob.alphas = [1e-13]

# setting tolerances
prob.tol = 1e-5
prob.stol = 1e-8

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)

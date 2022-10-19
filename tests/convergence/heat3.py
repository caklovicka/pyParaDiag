# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../..')
import numpy as np

from problem_examples_parallel.heat_2d_pbc_central6 import Heat
prob = Heat()

# choosing a number of points
prob.spatial_points = [300, 300]
prob.time_points = 3

# choosing a time domain
prob.T_start = np.pi
prob.T_end = prob.T_start + 0.1

# choosing the number of intervals handled in parallel
prob.time_intervals = 16
prob.rolling = 1

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'

# setting maximum number of iterations
prob.maxiter = 3
prob.smaxiter = 1000

# choosing a setting for the alpha sequence
prob.optimal_alphas = False
prob.alphas = [1e-12]

# setting tolerances
prob.tol = 1e-12
prob.stol = 1e-13

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)

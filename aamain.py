# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from examples.nonlinear.boltzmann_3d_pbc_upwind1 import Boltzmann
prob = Boltzmann()

# choosing a number of points
prob.spatial_points = [20, 30, 30, 30]
prob.time_points = 3

# choosing a time domain
prob.T_start = 0

# choosing the number of intervals handled in parallel
prob.time_intervals = 1
prob.rolling = 4

prob.T_end = 1e-3 * prob.rolling * prob.time_intervals

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'gmres'

# setting maximum number of iterations
prob.maxiter = 10
prob.smaxiter = 50

# choosing a setting for the alpha sequence
prob.alphas = [1e-8]

# setting tolerances
prob.tol = 1e-5
prob.stol = 1e-8

prob.setup()
prob.solve()
prob.summary(details=True)

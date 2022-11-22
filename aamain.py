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

# tmp
prob.document = 'None'
prob.betas = [0]

# choosing a number of points
prob.spatial_points = [100, 50, 30, 30]
prob.time_points = 1

# choosing a time domain
prob.T_start = 0

# choosing the number of intervals handled in parallel
prob.time_intervals = 1
prob.rolling = 32

prob.T_end = 1e-4 * prob.time_intervals * prob.rolling

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'

# setting maximum number of iterations
prob.maxiter = 20
prob.smaxiter = 500

# choosing a setting for the alpha sequence
prob.alphas = [1e-6]

# setting tolerances
prob.tol = 1e-6
prob.stol = 1e-8

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)

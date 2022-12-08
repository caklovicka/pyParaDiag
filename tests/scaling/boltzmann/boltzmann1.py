# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../../..') # for core
sys.path.append('../../../../../../..')     # for jube

from examples.nonlinear.boltzmann_3d_pbc_upwind1 import Boltzmann
prob = Boltzmann()

# choosing a number of points
prob.spatial_points = [384, 72, 36, 36]
prob.time_points = 1
prob.knudsen = 1e-2

# choosing a time domain
prob.T_start = 0

# choosing the number of intervals handled in parallel
#prob.time_intervals = 1
#prob.rolling = 32

prob.T_end = 0.001 * prob.rolling * prob.time_intervals

# choosing a parallelization strategy
prob.proc_col = 32
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'

# setting maximum number of iterations
prob.maxiter = 10
prob.smaxiter = 50

# choosing a setting for the alpha sequence
prob.alphas = [1e-8]

# setting tolerances
prob.tol = 1e-4
prob.stol = 1e-6

prob.setup()
prob.solve()
prob.summary(details=True)

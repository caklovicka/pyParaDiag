# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# RUNTIME ARGS
# prob.proc_col

import sys
sys.path.append('../../..')                 # for core
sys.path.append('../../../../../../..')     # for jube

from examples.nonlinear.boltzmann_3d_pbc_upwind1 import Boltzmann
prob = Boltzmann()

# choosing a number of points
prob.spatial_points = [128, 64, 32, 32]
prob.time_points = 1
prob.knudsen = 1e-2

prob.rolling = 32
prob.time_intervals = 1

# choosing a time domain
prob.T_start = 0
prob.T_end = 0.004 * prob.rolling * prob.time_intervals
# works with dt = 0.007 and M = 2, L = 20, but not M = 1
# dt = 0.006, L = 32, M = 1 blows up after 7 steps
# dt = 0.005, L = 32, M = 1 blows up
# dt = 0.004, L = 32, M = 1 works? (set maxiter higher)

# choosing a parallelization strategy
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'

# setting maximum number of iterations
prob.maxiter = 5
prob.smaxiter = 50

# choosing a setting for the alpha sequence
prob.alphas = [1e-8]

# setting tolerances
prob.tol = 1e-5
prob.stol = 1e-6

prob.setup()
prob.solve()
prob.summary(details=True)

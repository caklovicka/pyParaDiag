# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from examples.nonlinear.boltzmann_3d_pbc_upwind1 import Boltzmann
prob = Boltzmann()

# choosing a number of points
prob.spatial_points = [100, 50, 30, 30]            # number of unknowns for the 2D spatial problem
prob.time_points = 1                        # number of collocation nodes (Gauss-Radau-Right)

# choosing a time domain
prob.T_start = 0

# choosing the number of intervals handled in parallel
prob.time_intervals = 4
prob.rolling = 1

prob.T_end = 1e-2 * prob.time_intervals * prob.rolling

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'

# setting maximum number of iterations
prob.maxiter = 5
prob.smaxiter = 100

# choosing a setting for the alpha sequence
prob.alphas = [1e-6]

# setting tolerances
prob.tol = 1e-3
prob.stol = 1e-4

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)

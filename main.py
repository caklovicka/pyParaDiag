# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from problem_examples_parallel.advection_2d_pbc_upwind5 import Advection
prob = Advection()

# choosing a number of points
prob.spatial_points = [60, 60]
prob.time_points = 1

# choosing a time domain
prob.T_start = 0
prob.T_end = 0.0128

# choosing the number of intervals handled in parallel
prob.time_intervals = 4
prob.rolling = 1

# choosing a parallelization strategy
prob.proc_col = 2
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'                      # custom (defined in the problem class through linear_solver), lu or gmres (from scipy)

# setting maximum number of iterations
prob.maxiter = 10                           # number of Paralpha maxiters
prob.smaxiter = 50                          # number of inner solver maxiters

# choosing a setting for the alpha sequence
prob.optimal_alphas = True
#prob.alphas = [1e-5, 1e-2, 0.1]

# setting tolerances
prob.tol = 1e-12                            # a stopping tolerance for Paralpha
prob.stol = 1e-16                           # a stopping relative tolerance for the inner solver
prob.m0 = 10 * (prob.T_end - prob.T_start)  # a starting choice for the m_k sequence

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)

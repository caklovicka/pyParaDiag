# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# todo make sequential sun supported for increment

from problem_examples.linear.advection_2d_pbc_upwind5 import Advection
prob = Advection()

# choosing a number of points
prob.spatial_points = [700, 600]            # number of unknowns for the 2D spatial problem
prob.time_points = 1                        # number of collocation nodes (Gauss-Radau-Right)

# choosing a time domain
prob.T_start = 0
prob.T_end = 0.0128 / 10

# choosing the number of intervals handled in parallel
prob.time_intervals = 1                    # number of time-steps Paralpha will compute in parallel, for now needs to be a power of 2
prob.rolling = 4                            # number of Paralpha propagations in a classical/sequential sense

# choosing a parallelization strategy
prob.proc_col = 2                           # number of cores handling the collocation problem
prob.proc_row = prob.time_intervals         # number of cores handling time-steps. For now it has to be the same as number of time_intervals

# choosing a solver
prob.solver = 'custom'                      # custom (defined in the problem class through linear_solver), lu or gmres (from scipy)

# setting maximum number of iterations
prob.maxiter = 5                           # number of Paralpha maxiters
prob.smaxiter = 100                          # number of inner solver maxiters

# choosing a setting for the alpha sequence
prob.alphas = [1e-3]

# setting tolerances
prob.tol = 0#1e-12                            # a stopping tolerance for Paralpha
prob.stol = 1e-16                           # a stopping relative tolerance for the inner solver
prob.m0 = 10 * (prob.T_end - prob.T_start)  # a starting choice for the m_k sequence

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
#prob.summary(details=True)
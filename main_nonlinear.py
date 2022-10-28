# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# todo support sequential run

from problem_examples.nonlinear.allen_cahn_2d_pbc_central2 import AllenCahn
prob = AllenCahn()

# choosing a number of points
prob.spatial_points = [100, 100]
prob.time_points = 1
prob.eps = 0.1

# choosing a time domain
prob.T_start = 0

# choosing the number of intervals handled in parallel
prob.time_intervals = 1
prob.rolling = 4
prob.T_end = prob.rolling * prob.time_intervals * prob.eps ** 2 / 2

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'

# setting maximum number of iterations
prob.maxiter = 30
prob.smaxiter = 500

# choosing a setting for the alpha sequence
prob.alphas = [1e-3]
prob.betas = [1]

# setting tolerances
prob.tol = 1e-7
prob.stol = 1e-9

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)


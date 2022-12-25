# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# RUNTIME ARGS
# prob.proc_col
# prob.rolling
# prob.time_intervals

import sys
sys.path.append('../../..')                 # for core
sys.path.append('../../../../../../..')     # for jube

from examples.nonlinear.boltzmann_3d_pbc_upwind1 import Boltzmann
prob = Boltzmann()

# choosing a number of points
prob.spatial_points = [384, 72, 36, 36]
prob.time_points = 3
prob.knudsen = 1e-2

# choosing a time domain
prob.T_start = 0
#prob.time_intervals = 8
prob.rolling = 1
prob.T_end = 0.2 * prob.rolling * prob.time_intervals

# choosing a parallelization strategy
prob.proc_row = 3
prob.proc_col = 1

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

prob.comm.Barrier()
if prob.rank == prob.size - 1:
    up = prob.u_loc[-prob.global_size_A:]
    f = open('exact3.txt', 'w')
    f.write(up)
    f.close()

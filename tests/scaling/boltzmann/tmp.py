# the following lines disable the numpy multithreading [optional]
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "1"

import sys
sys.path.append('../../..')                  # for core
sys.path.append('../../../../../../..')      # for jube

from examples.nonlinear.boltzmann_1x3v_pbc_upwind1_python import Boltzmann
#from examples.nonlinear.boltzmann_3d_pbc_upwind1_Qpython import Boltzmann
prob = Boltzmann()

# RUNTIME ARGS

# choosing a number of points
prob.spatial_points = [20, 10, 10, 10]
prob.time_points = 1
prob.knudsen = 1e-2
prob.proc_col = 1
prob.rolling = 1
prob.time_intervals = 1

# choosing a time domain
prob.T_start = 0
prob.T_end = 1e-3 * prob.rolling * prob.time_intervals

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
prob.tol = 1e-4
prob.stol = 1e-6
prob.document = 'tmp.out'

prob.setup()
plt.show()
prob.solve()
prob.summary(details=True)
prob.__write_u_last_in_txt__(type=float)

plt.plot(np.transpose(prob.u0_loc.reshape(prob.spatial_points), axes=(3, 2, 1, 0))[:, 0, 0, 0])
plt.plot((prob.u0_loc.reshape(prob.spatial_points)[:, 0, 0, 0]))
plt.show()


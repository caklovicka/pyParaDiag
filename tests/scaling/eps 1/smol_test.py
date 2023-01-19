# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../../..')                 # for core
sys.path.append('../../../../../../..')     # for jube
import numpy as np

from examples.nonlinear.allen_cahn_2d_pbc_central6 import AllenCahn
prob = AllenCahn()
prob.spatial_points = [3, 3]
prob.time_points = 2
prob.tol = 1e-13
prob.stol = 1e-15
prob.T_end = 0.4

#prob.time_intervals = 2
#prob.rolling = 1
prob.betas = [0]

prob.eps = 1
prob.T_start = 0
#prob.proc_col = 1
prob.solver = 'gmres'
prob.maxiter = 10
prob.smaxiter = 500
prob.alphas = [1e-8]
prob.R = 1
prob.X_left = -2 * prob.R
prob.X_right = 2 * prob.R
prob.Y_left = -2 * prob.R
prob.Y_right = 2 * prob.R

prob.proc_row = prob.time_intervals

prob.setup()
prob.solve()
prob.summary(details=False)

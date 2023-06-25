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

# time steps: 24
# rolling from runtime
# time_intervals from runtime

from examples.nonlinear.allen_cahn_2d_pbc_central2 import AllenCahn
prob = AllenCahn()
prob.spatial_points = [256, 256]
prob.time_points = 4
prob.tol = 1e-10
prob.stol = 1e-11
prob.T_end = 0.024

prob.betas = [1]
prob.T_start = 0
prob.proc_col = 4
prob.solver = 'gmres'
prob.maxiter = 50
prob.smaxiter = 1000
prob.alphas = [1e-4]

prob.proc_row = prob.time_intervals

prob.setup()
prob.solve()
prob.summary(details=False)

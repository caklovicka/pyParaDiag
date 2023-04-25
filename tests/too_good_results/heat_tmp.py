import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../..')                 # for core
sys.path.append('../../../../../..')     # for jube
import numpy as np
from examples.linear.heat_2d_pbc_central6 import Heat

prob = Heat()
N = 350
prob.spatial_points = [N, N]
prob.tol = 1e-12
prob.proc_col = 1
prob.time_points = 3
prob.T_start = np.pi
prob.T_end = np.pi + 0.16
prob.solver = 'custom'
prob.maxiter = 3
prob.smaxiter = 50
prob.stol = 1e-13
prob.optimal_alphas = False
prob.alphas = [1e-8]
prob.rolling = 1
prob.time_intervals = 64
prob.proc_row = 64

prob.setup()
prob.solve()
prob.summary(details=True)
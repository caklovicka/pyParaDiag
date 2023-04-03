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
from examples.linear.heat_2d_pbc_central2_steady_state import Heat

prob = Heat()
N = 350
prob.spatial_points = [N, N]
prob.tol = 1e-5
prob.proc_col = 1
prob.time_points = 1
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 0.32
prob.solver = 'custom'
prob.maxiter = 10
prob.smaxiter = 50
prob.stol = 1e-6
prob.optimal_alphas = False
prob.alphas = [1e-2]

prob.setup()
prob.solve()
prob.summary(details=True)
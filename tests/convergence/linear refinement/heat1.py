import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../../..')
import numpy as np
from examples.linear.heat_2d_pbc_central2 import Heat as Heat2

prob = Heat2()
N = 350
prob.spatial_points = [N, N]
prob.tol = 1e-5
prob.proc_col = 1
prob.time_points = 1
prob.optimal_alphas = True
prob.T_start = np.pi
prob.T_end = prob.T_start + 0.32
prob.solver = 'custom'
prob.maxiter = 10
prob.smaxiter = 50
prob.stol = 1e-6
prob.m0 = (prob.T_end - prob.T_start)/prob.rolling
prob.time_intervals = 64
prob.proc_row = 64

prob.setup()
prob.solve()
prob.summary(details=True)

if prob.rank == prob.size - 1:
    exact = prob.u_exact(prob.T_end, prob.x).flatten()[prob.row_beg:prob.row_end]
    approx = prob.u_last_loc.flatten()
    d = exact - approx
    d = d.flatten()
    err_abs = np.linalg.norm(d, np.inf)
    print('abs err = {}'.format(err_abs))
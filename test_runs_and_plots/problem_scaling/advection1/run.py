import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../../../../../../..')    # for jube
import numpy as np
from problem_examples_parallel.advection_2d_pbc_upwind2 import Advection

prob = Advection()
N = 350
prob.spatial_points = [N, N]
prob.tol = 1e-5
prob.proc_col = 1
prob.time_points = 1
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 0.003
prob.solver = 'custom'
prob.maxiter = 5
prob.smaxiter = 50
prob.stol = 1e-8
prob.m0 = 1 * (prob.T_end - prob.T_start)

prob.setup()
prob.solve()
prob.summary(details=True)

if prob.rank == prob.size - 1:
    exact = prob.u_exact(prob.T_end, prob.x).flatten()[prob.row_beg:prob.row_end]
    approx = prob.u_last_loc.flatten()
    d = exact - approx
    d = d.flatten()
    err_abs = np.linalg.norm(d, np.inf)
    print('abs err  = {}'.format(err_abs))



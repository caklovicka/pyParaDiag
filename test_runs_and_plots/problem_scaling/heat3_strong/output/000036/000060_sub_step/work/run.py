import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../../../../../../..')    # for jube
sys.path.append('../../../../../../../..')    # for pySDC
import numpy as np
from mpi4py import MPI
from problem_examples_parallel.heat_2d_pbc_central6 import Heat as Heat6

prob = Heat6()
N = 350
prob.spatial_points = [N, N]
prob.tol = 1e-12
prob.proc_col = 3
prob.time_points = 3
prob.optimal_alphas = True
prob.T_start = np.pi
prob.T_end = prob.T_start + 0.16
prob.solver = 'custom'
prob.maxiter = 10
prob.smaxiter = 50
prob.stol = 1e-13
prob.m0 = (prob.T_end - prob.T_start)/prob.rolling


prob.setup()
prob.solve()
prob.summary(details=True)

if prob.rank <= prob.size - prob.size_subcol_seq:
    exact = prob.u_exact(prob.T_end, prob.x).flatten()[prob.row_beg:prob.row_end]
    approx = prob.u_last_loc.flatten()
    d = exact - approx
    d = d.flatten()
    err_abs = np.linalg.norm(d, np.inf)
    err_abs_root = prob.comm_subcol_seq.reduce(err_abs, op=MPI.MAX, root=prob.size_subcol_seq - 1)
    if prob.rank == prob.size - 1:
        print('abs err = {}'.format(err_abs_root))


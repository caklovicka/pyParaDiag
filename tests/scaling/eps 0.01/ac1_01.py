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

# time steps: 64
# rolling from runtime
# time_intervals from runtime
# beta from runtime

from examples.nonlinear.allen_cahn_2d_pbc_central2_Ae import AllenCahn
prob = AllenCahn()
prob.spatial_points = [320, 320]
prob.time_points = 2
prob.tol = 1e-5
prob.stol = 1e-7

prob.eps = 0.01
prob.T_start = 0
prob.T_end = 0.003
prob.proc_col = 1
prob.solver = 'custom'
prob.maxiter = 50
prob.smaxiter = 500
prob.alphas = [1e-8]

prob.proc_row = prob.time_intervals

prob.setup()
prob.solve()
prob.summary(details=False)

prob.comm.Barrier()
if prob.rank == prob.size - 1:
    up = prob.u_loc[-prob.global_size_A:]
    us = np.empty_like(up)
    f = open('../../../../exact1.txt', 'r')
    lines = f.readlines()
    for i in range(prob.global_size_A):
        us[i] = complex(lines[i])
    f.close()
    diff = us - up
    print('diff =', np.linalg.norm(diff, np.inf), flush=True)

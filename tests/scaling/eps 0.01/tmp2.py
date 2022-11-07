# the following lines disable the numpy multithreading [optional]
import os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../../..')
sys.path.append('../../../../../../..')    # for jube

# time steps: 128
# rolling from runtime
# time_intervals from runtime
# beta from runtime

from examples.nonlinear.allen_cahn_2d_pbc_central2 import AllenCahn
prob = AllenCahn()
prob.spatial_points = [100, 100]
prob.time_points = 2
prob.tol = 1e-8
prob.stol = 1e-10

prob.eps = 0.01
prob.T_start = 0
prob.T_end = 0.0075 / 128 * 16
prob.proc_col = 1
prob.solver = 'gmres'
prob.maxiter = 50
prob.smaxiter = 100
prob.alphas = [1e-8]

prob.proc_row = prob.time_intervals

prob.setup()
prob.solve()
prob.summary(details=False)

prob.comm.Barrier()

if prob.rank == prob.size - 1:
    up = prob.u_loc[-prob.global_size_A:]
    us = np.empty_like(up)
    f = open('exact.txt', 'r')
    lines = f.readlines()
    for i in range(prob.global_size_A):
        us[i] = complex(lines[i])
    f.close()
    diff = us - up
    print(np.linalg.norm(diff, np.inf))

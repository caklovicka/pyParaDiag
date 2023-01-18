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

# time steps: 128
# rolling from runtime
# time_intervals from runtime
# beta from runtime

from examples.nonlinear.allen_cahn_2d_pbc_central6 import AllenCahn
prob = AllenCahn()
prob.spatial_points = [400, 400]
prob.time_points = 3
prob.tol = 1e-13
prob.stol = 1e-15
prob.T_end = 0.4

prob.eps = 1
prob.T_start = 0
prob.proc_col = 3
prob.solver = 'gmres'
prob.maxiter = 50
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

'''
f = open('exact3.txt', 'w')
up = prob.u_loc[-prob.global_size_A:]
for i in range(prob.global_size_A):
    f.write(str(up[i]) + '\n')
f.close()

if prob.rank == prob.size - 1:
    up = prob.u_loc[-prob.global_size_A:]
    us = np.empty_like(up)
    f = open('exact3.txt', 'r')
    lines = f.readlines()
    for i in range(prob.global_size_A):
        us[i] = complex(lines[i])
    f.close()
    diff = us - up
    print(np.linalg.norm(diff, np.inf))
    '''

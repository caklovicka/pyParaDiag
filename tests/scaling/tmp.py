# the following lines disable the numpy multithreading [optional]
import os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../..')
sys.path.append('../../../../../..')    # for jube

# time steps: 128
# rolling from runtime
# time_intervals from runtime
# beta from runtime

from examples.nonlinear.tmp import AllenCahn
prob = AllenCahn()
prob.spatial_points = [100, 100]
prob.time_points = 1
prob.tol = 1e-8
prob.stol = 1e-10
prob.rolling = 1
prob.time_intervals = 64

prob.time_points = 1
prob.eps = 0.01
prob.T_start = 0
prob.T_end = 0.1 * prob.eps ** 3 * prob.rolling * prob.time_intervals
prob.proc_col = 1
prob.solver = 'gmres'
prob.maxiter = 50
prob.smaxiter = 100
prob.alphas = [1e-12]

prob.proc_row = prob.time_intervals

prob.setup()
prob.solve()
prob.summary(details=False)


# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('../../..')

# time steps: 512
# rolling from runtime
# time_intervals from runtime
# beta from runtime

from examples.nonlinear.allen_cahn_2d_pbc_central4 import AllenCahn
prob = AllenCahn()
prob.spatial_points = [128, 128]
prob.time_points = 3
prob.tol = 1e-9
prob.stol = 1e-11

prob.eps = 0.01
prob.T_start = 0
prob.T_end = 0.03
prob.proc_col = 1
prob.solver = 'gmres'
prob.maxiter = 50
prob.smaxiter = 500
prob.alphas = [1e-8]

prob.proc_row = prob.time_intervals

prob.setup()
print(prob.dx[0]**4, prob.dt**(2*prob.time_points - 1), prob.dt < prob.eps ** 2)
print(prob.eps, prob.eps**2, prob.eps**3, prob.eps**4)
prob.solve()
prob.summary(details=False)
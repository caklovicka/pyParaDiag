# the following lines disable the numpy multithreading [optional]
import os
import numpy as np
np.set_printoptions(linewidth=np.inf)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from examples.heat_2d_pbc_central2 import Heat
prob = Heat()

# choosing a number of points
prob.spatial_points = [4, 4]
prob.time_points = 2
prob.T_start = 0
prob.T_end = 1e-5
prob.time_intervals = 4
#prob.proc_col = 2
prob.proc_row = prob.time_intervals
prob.solver = 'custom'
prob.maxiter = 5
prob.smaxiter = 100
prob.alphas = [1e-3]
prob.tol = 1e-12
prob.stol = 1e-16

prob.setup()
if prob.rank_global == 0:
    print(prob.yd(0, prob.x))
prob.solve()
#prob.summary(details=True)

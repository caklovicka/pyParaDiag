from problem_examples_parallel.advection_2d_central2 import Advection
from problem_examples_parallel.advection_2d_pbc_central4 import Advection as Advection4
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
prob = Advection4()
N = 1000
prob.spatial_points = [N, N]
prob.tol = 1e-11
prob.proc_col = 72
prob.proc_row = 1
prob.time_intervals = 1
prob.rolling = 1
prob.time_points = 3
prob.optimal_alphas = True
prob.T_start = 0
prob.T_end = 0.009554572093283932
prob.solver = 'custom'
prob.maxiter = 5
prob.smaxiter = 50
prob.stol = 1e-15
prob.m0 = 10 * (prob.T_end - prob.T_start)

prob.setup()
prob.solve()
prob.summary(details=True)



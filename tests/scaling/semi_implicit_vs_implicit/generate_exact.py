import sys
sys.path.append('../../..')                 # for core
sys.path.append('../../../../../../..')     # for jube
import numpy as np

# time steps: 64
# rolling from runtime
# time_intervals from runtime

from examples.nonlinear.allen_cahn_2d_pbc_central6 import AllenCahn
prob = AllenCahn()
prob.spatial_points = [200, 200]
prob.tol = 1e-8
prob.stol = 1e-10

prob.time_points = 2
prob.eps = 0.01
prob.T_start = 0
prob.proc_col = 1
prob.solver = 'gmres'
prob.maxiter = 15
prob.smaxiter = 500
prob.alphas = [1e-8]
prob.T_end = 1e-2
prob.betas = [0]

prob.proc_row = prob.time_intervals

prob.setup()
print('tol * eps^2 = ', prob.tol * prob.eps ** 2)
print(prob.dx[0]**4, prob.dt**(2 * prob.time_points - 1))
prob.solve()
prob.summary(details=False)

f = open('exact.txt', 'w')
up = prob.u_loc[-prob.global_size_A:]
for i in range(prob.global_size_A):
    f.write(str(up[i]) + '\n')
f.close()

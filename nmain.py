# the following lines disable the numpy multithreading [optional]
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from problem_examples.nonlinear.allen_cahn_2d_pbc_central2 import AllenCahn
prob = AllenCahn()

# choosing a number of points
prob.spatial_points = [100, 100]
prob.time_points = 3
prob.eps = 0.04

# choosing a time domain
prob.T_start = 0

# choosing the number of intervals handled in parallel
prob.time_intervals = 1
prob.rolling = 32
prob.T_end = 4.6e-4

# choosing a parallelization strategy
prob.proc_col = 1
prob.proc_row = prob.time_intervals

# choosing a solver
prob.solver = 'custom'

# setting maximum number of iterations
prob.maxiter = 10
prob.smaxiter = 500

# choosing a setting for the alpha sequence
prob.alphas = [1e-2]
prob.betas = [0]

# setting tolerances
prob.tol = 1e-5
prob.stol = 1e-7

prob.setup()                                # must be before solve()
prob.solve()                                # this is where magic happens
prob.summary(details=True)


prob.document = 'exact.txt'
prob.__write_u_last_in_txt__()


# CHECK OUTPUT
prob.__fill_u_last__(fill_old=False)
uexact = np.empty_like(prob.u_last_loc, dtype=complex)

if prob.frac > 1:
    for proc in range(prob.size - prob.size_subcol_seq, prob.size, 1):
        if proc == prob.rank:
            uexact = np.empty_like(prob.u_last_loc, dtype=complex)
            file = open('exact.txt', 'r')

            k = 0
            lines = file.readlines()
            for i in range(prob.row_beg, prob.row_end, 1):
                if lines[i] == '\n':
                    break
                uexact[k] = complex(lines[i])
                k += 1
            file.close()
            print(prob.rank, prob.norm(prob.u_last_loc - uexact))
else:
    if prob.rank == prob.size - 1:
        uexact = np.empty_like(prob.u_last_loc, dtype=complex)
        file = open('exact.txt', 'r')

        k = 0
        lines = file.readlines()
        for i in range(prob.row_beg, prob.row_end, 1):
            if lines[i] == '\n':
                break
            uexact[k] = complex(lines[i])
            k += 1
        file.close()
        print(prob.rank, prob.norm(prob.u_last_loc - uexact))

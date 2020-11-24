from mpi4py import MPI

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
time = 0
k = 50

vec = np.array(400 * 400)
for i in range(k):
    time_beg = MPI.Wtime()
    out = comm.allreduce(vec, op=MPI.SUM)
    time += (MPI.Wtime() - time_beg)
    vec = 1.2 * vec

tot_time = comm.reduce(time, op=MPI.SUM, root=0)
if rank == 0:
    print(tot_time)

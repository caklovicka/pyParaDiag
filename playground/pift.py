#######################################################
# testing parallel fft and ifft with reversed butterfly
#######################################################

from mpi4py import MPI
import numpy as np
import argparse

# ------------- main -----------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10    # size of vector
L = size
u = np.random.rand(N) + 1j * np.random.rand(N)   # generate the array
u_original = u.copy()
n = int(np.log2(L))
P = format(rank, 'b').zfill(n)     # binary of the rank in string
R = P[::-1]          # reversed binary in string

# ----------------- fft --------------------
w = np.exp(-2 * np.pi * 1j / L)

for k in range(n):
    p = L // 2**(k + 1)
    r = int(R, 2) % 2**(k + 1) - 2**k
    scalar = w**(r * p)
    factor = 1

    if P[k] == '1':
        factor = -1
        if scalar != 1:     # multiply if the factor is != 1
            u *= scalar

    # make a new string and an int from it, a proc to communicate with
    comm_with = list(P)
    if comm_with[k] == '1':
        comm_with[k] = '0'
    else:
        comm_with[k] = '1'
    comm_with = int(''.join(comm_with), 2)

    # now communicate
    req = comm.isend(u, dest=comm_with, tag=k)
    ur = comm.recv(source=comm_with, tag=k)
    req.Wait()

    # glue the info
    u = ur + factor * u

# ------- end of fft -----------
# now we have permuted array that is transformed.
# ---------- ifft ----------
w = np.exp(2 * np.pi * 1j / L)

u /= L
for k in range(n):
    p = L // 2**(n-k)
    r = int(R, 2) % 2**(n-k) - 2**(n - k - 1)
    scalar = w ** (r * p)
    factor = 1

    if R[k] == '1':
        factor = -1

    # make a new string and an int from it, a proc to communicate with
    comm_with = list(R)
    if comm_with[k] == '1':
        comm_with[k] = '0'
    else:
        comm_with[k] = '1'
    comm_with = int(''.join(comm_with)[::-1], 2)

    # now communicate
    req = comm.isend(u, dest=comm_with, tag=k)
    ur = comm.recv(source=comm_with, tag=k)
    req.Wait()

    # glue the info
    u = ur + factor * u

    # scale the output
    if R[k] == '1' and scalar != 1:
        u *= scalar

# ------ end ifft -------
#-------- check ---------

err = np.linalg.norm(u_original - u)
err_max = comm.reduce(err, op=MPI.MAX, root=0)
if rank == 0:
    print(err_max)



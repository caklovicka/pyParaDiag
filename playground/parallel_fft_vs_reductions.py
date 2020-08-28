#####################################################
# testing parallel forward fft scaling and reductions
#####################################################

from mpi4py import MPI
import numpy as np
import argparse

# ------------- main -----------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=10)
args = parser.parse_args().__dict__

N = args['N']    # size of vector
L = size
u = np.random.rand(N) + 1j * np.random.rand(N)   # generate the array
n = int(np.log2(L))
binary = format(rank, 'b').zfill(n)     # binary of the rank in string
index = binary[::-1]                    # reversed binary, the one we need to get
rank_swap = int(index, 2)

# --------- forward fft ---------
comm.barrier()
time_start = MPI.Wtime()

# swap USE NONBLOCKING HERE
if rank_swap != rank:
    req = comm.isend(u, dest=rank_swap, tag=0)
    U = comm.recv(source=rank_swap, tag=0)  # this will be the transformation
    req.Wait()
else:
    U = u.copy()

w = np.exp(-2 * np.pi * 1j / L)

# stages of butterfly
for k in range(n):
    p = int(L / 2**(k + 1))     # twiddle factor or whatever the name is
    r = int(rank % 2**(k + 1) - 2**k)   # exponent of w
    factor = 1      # + or - factor
    if index[k] == '1':     # values that need multiplying
        U *= w**(p*r)
        factor = -1

    # figure out with whom to communicate
    communicate_with = list(index)
    if communicate_with[k] == '1':
        communicate_with[k] = '0'
    else:
        communicate_with[k] = '1'
    communicate_with = ''.join(communicate_with)
    communicate_with = int(communicate_with[::-1], 2)

    # now communicate
    req = comm.isend(U, dest=communicate_with, tag=k+1)
    Ur = comm.recv(source=communicate_with, tag=k+1)
    req.Wait()
    U = Ur + factor * U

'''
# check with numpy on root
u_all = np.array(comm.gather(u, root=0))
U_all = np.array(comm.gather(U, root=0))
if rank == 0:
    U_seq = np.fft.fft(u_all, axis=0)
    err = np.linalg.norm(U_seq - U_all, np.inf)
    print(err)
'''
time = MPI.Wtime() - time_start
time_radix = comm.reduce(time, op=MPI.MAX, root=0)
#U_radix = np.array(comm.gather(U, root=0))

# end radix, start the other way
comm.barrier()
time_start = MPI.Wtime()

for i in range(size):
    u_send = w**(i * rank) * u
    if rank == i:
        U = comm.reduce(u_send, op=MPI.SUM, root=i)
    else:
        comm.reduce(u_send, op=MPI.SUM, root=i)

time = MPI.Wtime() - time_start
time_reductions = comm.reduce(time, op=MPI.MAX, root=0)
#U_reductions = np.array(comm.gather(U, root=0))

if rank == 0:
    #err = np.linalg.norm(U_radix - U_reductions)
    #print('err = ' + str(err))
    print('array size = {}, nproc = {}\ntime_radix = {}\ntime_reductions = {}'.format(N, L, time_radix, time_reductions))




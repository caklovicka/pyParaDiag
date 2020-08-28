##################################################
# testing parallel forward ifft with a serial check
##################################################

from mpi4py import MPI
import numpy as np

# ------------- main -----------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10    # size of vector
L = size
U = np.random.rand(N) + 1j * np.random.rand(N)   # generate the array
U_original = U.copy()
n = int(np.log2(L))
binary = format(rank, 'b').zfill(n)     # binary of the rank in string
rank_swap = int(binary[::-1], 2)    # reversed binary, the one we need to get

# --------- backward fft ---------
w = np.exp(2 * np.pi * 1j / L)
U /= L

# stages of butterfly
for k in range(n):
    p = int(L / 2**(n - k))     # twiddle factor or whatever the name is
    r = int(rank % 2**(n-k) - 2**(n - k - 1))   # exponent of w
    factor = 1      # + or - factor
    if binary[k] == '1':     # values that need multiplying
        factor = -1

    # figure out with whom to communicate
    communicate_with = list(binary)
    if communicate_with[k] == '1':
        communicate_with[k] = '0'
    else:
        communicate_with[k] = '1'
    communicate_with = ''.join(communicate_with)
    communicate_with = int(communicate_with, 2)

    # now communicate
    req = comm.isend(U, dest=communicate_with, tag=k+1)
    Ur = comm.recv(source=communicate_with, tag=k+1)
    req.Wait()
    U = Ur + factor * U
    if factor == -1:
        U *= w ** (p * r)

# swap USE NONBLOCKING HERE
if rank_swap != rank:
    req = comm.isend(U, dest=rank_swap, tag=0)
    u = comm.recv(source=rank_swap, tag=0)  # this will be the transformation
    req.Wait()
else:
    u = U.copy()

# check with numpy on root
u_all = np.array(comm.gather(u, root=0))
U_all = np.array(comm.gather(U_original, root=0))
if rank == 0:
    u_seq = np.fft.ifft(U_all, axis=0)
    err = np.linalg.norm(u_seq - u_all, np.inf)
    print(err)

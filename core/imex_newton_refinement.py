import numpy as np
from mpi4py import MPI
from core.helpers import Helpers
import os
import matplotlib.pyplot as plt
import sys
np.set_printoptions(linewidth=np.inf, precision=5, threshold=sys.maxsize)

class PartiallyCoupled(Helpers):

    def __init__(self):

        super().__init__()
        self.setup_var = False

    def setup(self):

        self.setup_var = True
        super().setup()

        if self.time_intervals == 1:
            # TODO to support the sequential run
            self.alpha = 0

        self.residual = []
        self.convergence = 1

    def solve(self):

        self.comm.Barrier()
        time_beg = MPI.Wtime()

        h0 = np.zeros(self.rows_loc, dtype=complex, order='C')  # initial guess for inner systems
        self.stop = False

        self.__fill_initial_guesses__()
        if self.time_points == 1:
            v_loc = self.__get_v_Euler__()
        else:
            raise RuntimeError('Not implemented for M > 1')

        while not self.stop:       # main iterations

            # compute residual
            if self.time_points == 1:
                res_loc = self.__get_linear_residual_Euler__(v_loc)
            else:
                raise RuntimeError('Not implemented for M > 1')

            '''
            for r in range(self.size_global):
                self.comm_global.Barrier()
                if r == self.rank_global:
                    print(self.rank, self.rank_global, 'res_loc = ', res_loc.real, flush=True)
                    self.comm_global.Barrier()
            '''

            res_norm = self.__get_max_norm__(res_loc)
            self.residual.append(res_norm)

            # if it did not converge for a given maximum iterations
            if self.iterations == self.maxiter and self.residual[-1] > self.tol:
                self.convergence = 0
                break

            # if the solution starts exploding, terminate earlier
            if self.residual[-1] > 1000:
                self.convergence = 0
                print('divergence? residual = ', self.residual[-1])
                break

            # do a parallel scaled FFT in time
            g_loc, Rev = self.__get_fft__(res_loc, self.alpha)

            # ------ PROCESSORS HAVE DIFFERENT INDICES ROM HERE! -------

            system_time = []
            its = []

            # TODO: different shifted matrices for state and adjoint
            exit()
            Zinv, D, Z, Cinv = self.__get_shifted_matrices__(int(Rev, 2), self.alpha)

            h_loc = self.__solve_substitution__(Zinv, g_loc)        # step 1 ... (Z x I) h = g

            time_solver = MPI.Wtime()

            h1_loc, it = self.__solve_inner_systems__(h_loc, D, h0.copy(), self.stol)
            system_time.append(MPI.Wtime() - time_solver)
            its.append(it)

            h_loc = self.__solve_substitution__(Z, h1_loc)      # step 3 ... (Zinv x I) h = h1
            if self.time_intervals > 1 or self.betas[i_beta] > 0:
                h1_loc = self.__solve_substitution__(Cinv, h_loc)  # step 4 ... (C x I) h1 = h
            else:
                self.u_loc = self.__solve_substitution__(Cinv, h_loc)

            self.system_time_max[rolling_interval].append(self.comm.allreduce(max(system_time), op=MPI.MAX))
            self.system_time_min[rolling_interval].append(self.comm.allreduce(min(system_time), op=MPI.MIN))
            self.solver_its_max[rolling_interval].append(self.comm.allreduce(max(its), op=MPI.MAX))
            self.solver_its_min[rolling_interval].append(self.comm.allreduce(min(its), op=MPI.MIN))
            self.inner_tols.append(self.stol)

            if self.time_intervals > 1 or self.betas[i_beta] > 0:
                h_loc = self.__get_ifft_h__(h1_loc, self.alphas[i_alpha])  # solving (Sinv x I) h1_loc = h with ifft
            else:  # to support the sequential run
                self.__get_ifft__(self.alphas[i_alpha])

            # ------ PROCESSORS HAVE NORMAL INDICES ROM HERE! -------

            self.iterations[rolling_interval] += 1
            if self.time_intervals > 1 or self.betas[i_beta] > 0:
                self.u_loc += h_loc     # update the solution
                self.consecutive_error[rolling_interval].append(self.__get_max_norm__(h_loc))  # consecutive error, error of the increment

            # end of main iterations (while loop)

            if rolling_interval + 1 < self.rolling:
                self.__fill_u0_loc__()

        max_time = MPI.Wtime() - time_beg
        self.algorithm_time = self.comm.allreduce(max_time, op=MPI.MAX)

        comm_time = self.communication_time
        self.communication_time = self.comm.allreduce(comm_time, op=MPI.MAX)

import numpy as np
from mpi4py import MPI
from core.helpers import Helpers
import os
import matplotlib.pyplot as plt


class PartiallyCoupled(Helpers):

    def __init__(self):

        super().__init__()
        self.setup_var = False

    def setup(self):

        self.setup_var = True
        super().setup()

        if self.time_intervals == 1:
            # TODO to support the sequential run
            self.u_last_loc = self.u0_loc.copy(order='C')
            self.alphas = [0]

        if self.betas == NotImplemented:
            self.betas = [0]

        self.residual = []
        self.convergence = 1

    def solve(self):

        self.comm.Barrier()
        time_beg = MPI.Wtime()
        h0 = np.zeros(self.rows_loc, dtype=complex, order='C')  # initial guess for inner systems

        for rolling_interval in range(self.rolling):

            self.stop = False
            self.consecutive_error.append([])
            self.residual.append([])
            self.consecutive_error[rolling_interval].append(np.inf)
            self.system_time_max.append([])
            self.system_time_min.append([])
            self.solver_its_max.append([])
            self.solver_its_min.append([])
            i_alpha = -1
            i_beta = -1
            t_start = self.T_start + self.time_intervals * rolling_interval * self.dt

            self.__fill_initial_u_loc__()
            v_loc = self.__get_v__(t_start)     # the rhs of the all-at-once system

            m0 = self.m0
            eps = np.finfo(complex).eps
            gamma = self.time_intervals * (3 * eps + self.stol)

            while not self.stop:       # main iterations

                i_alpha = self.__next_alpha__(i_alpha)
                i_beta = self.__next_beta__(i_beta)

                res_loc = self.__get_linear_residual__(v_loc)       # rhs vector of the iteration
                tmp = self.__get_F__()                             # add the explicit part
                res_loc += tmp
                res_norm = self.__get_max_norm__(res_loc)
                i_alpha = self.__next_alpha__(i_alpha)

                self.residual[rolling_interval].append(res_norm)
                if self.residual[rolling_interval][-1] <= self.tol or self.iterations[rolling_interval] == self.maxiter:
                    if self.residual[rolling_interval][-1] > self.tol:
                        self.convergence = 0
                    break

                if self.residual[rolling_interval][-1] > 1000:
                    self.convergence = 0
                    print('divergence? residual = ', self.residual[rolling_interval][-1])
                    break

                if self.time_intervals == 1 and self.betas[i_beta] == 0:
                    res_loc = self.__get_w__(self.alphas, v_loc, v_loc) + tmp

                g_loc, Rev = self.__get_fft__(res_loc, self.alphas[i_alpha])        # solving (S x I) g = w with ifft

                # ------ PROCESSORS HAVE DIFFERENT INDICES ROM HERE! -------

                system_time = []
                its = []

                Zinv, D, Z, Cinv = self.__get_shifted_matrices__(int(Rev, 2), self.alphas[i_alpha])

                h_loc = self.__solve_substitution__(Zinv, g_loc)        # step 1 ... (Z x I) h = g

                time_solver = MPI.Wtime()
                if self.betas[i_beta] > 0:
                    h1_loc, it = self.__solve_inner_systems_J__(h_loc, D, self.betas[i_beta], h0.copy(), self.stol)
                else:
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

import numpy as np
from mpi4py import MPI
from core.helpers import Helpers
import os

np.set_printoptions(precision=5, linewidth=np.inf)


class LinearIncrementParalpha(Helpers):

    def __init__(self):

        super().__init__()
        self.setup_var = False

    def setup(self):

        self.setup_var = True
        super().setup()

        if self.rolling > 1:
            self.u_last_loc = self.u0_loc.copy(order='C')
        self.consecutive_error = []
        self.residual = []
        if self.time_intervals == 1:    # to support the sequential run
            self.maxiter = 1
            self.alphas = [0]

    def solve(self):

        self.comm.Barrier()
        time_beg = MPI.Wtime()

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
            t_start = self.T_start + self.time_intervals * rolling_interval * self.dt
            h0 = np.zeros(self.rows_loc, dtype=complex, order='C')  # initial guess for inner systems

            v_loc = self.__get_v__(t_start)     # the rhs of the all-at-once system

            res_norm = None
            m0 = self.m0
            eps = np.finfo(complex).eps
            gamma = self.time_intervals * (3 * eps + self.stol)

            while self.iterations[rolling_interval] < self.maxiter and not self.stop:       # main iterations

                if self.time_intervals > 1:
                    res_loc = self.__get_linear_residual__(v_loc)      # rhs vector of the iteration
                    res_norm = self.__get_max_norm__(res_loc)
                    self.residual[rolling_interval].append(res_norm)
                else:   # to support the sequential run
                    res_loc = self.__get_w__(self.alphas[i_alpha], v_loc, v_loc)

                if self.time_points > 1 and self.residual[rolling_interval][-1] <= self.tol:
                    break

                if self.optimal_alphas is True:
                    self.alphas.append(np.sqrt((gamma * res_norm)/m0))
                    m0 = 2 * np.sqrt(gamma * m0 * res_norm)
                    if m0 <= self.tol:
                        self.stop = True
                i_alpha = self.__next_alpha__(i_alpha)

                g_loc, Rev = self.__get_fft__(res_loc, self.alphas[i_alpha])        # solving (S x I) g = w with ifft

                # ------ PROCESSORS HAVE DIFFERENT INDICES ROM HERE! -------

                system_time = []
                its = []

                Zinv, D, Z, Cinv = self.__get_shifted_matrices__(int(Rev, 2), self.alphas[i_alpha])

                h_loc = self.__solve_substitution__(Zinv, g_loc)        # step 1 ... (Z x I) h = g

                time_solver = MPI.Wtime()
                h1_loc, it = self.__solve_inner_systems__(h_loc, D, h0, self.stol)      # step 2 ... solve local systems (I - Di * A) h1 = h
                system_time.append(MPI.Wtime() - time_solver)
                its.append(it)

                h_loc = self.__solve_substitution__(Z, h1_loc)      # step 3 ... (Zinv x I) h = h1
                if self.time_intervals > 1:
                    h1_loc = self.__solve_substitution__(Cinv, h_loc)       # step 4 ... (C x I) h1 = h
                else:
                    self.u_loc = self.__solve_substitution__(Cinv, h_loc)

                self.system_time_max[rolling_interval].append(self.comm.allreduce(max(system_time), op=MPI.MAX))
                self.system_time_min[rolling_interval].append(self.comm.allreduce(min(system_time), op=MPI.MIN))
                self.solver_its_max[rolling_interval].append(self.comm.allreduce(max(its), op=MPI.MAX))
                self.solver_its_min[rolling_interval].append(self.comm.allreduce(min(its), op=MPI.MIN))
                self.inner_tols.append(self.stol)

                if self.time_intervals > 1:
                    h_loc = self.__get_ifft_h__(h1_loc, self.alphas[i_alpha])       # solving (Sinv x I) h1_loc = h with ifft
                else:   # to support the sequential run
                    self.__get_ifft__(self.alphas[i_alpha])

                # ------ PROCESSORS HAVE NORMAL INDICES ROM HERE! -------

                self.iterations[rolling_interval] += 1
                if self.time_intervals > 1:
                    self.u_loc += h_loc     # update the solution
                    self.consecutive_error[rolling_interval].append(self.__get_max_norm__(h_loc))  # consecutive error, error of the increment

                if self.iterations[rolling_interval] == self.maxiter:
                    self.stop = True

                self.__print_on_runtime__(t_start, rolling_interval)

                # end of main iterations (while loop)

            self.__write_u_in_txt__(rolling_interval)   # document writing

            if rolling_interval + 1 < self.rolling:     # update u0_loc (new initial condition) on processors that need it (first column) if we are not in the last rolling interval
                self.__fill_u_last__(fill_old=False)
                self.__bcast_u_last_loc__()

                if self.comm_last != MPI.COMM_NULL and self.time_intervals > 1:
                    self.u0_loc = self.u_last_loc.copy()

                # to support a sequential run
                elif self.time_intervals == 1:

                    if self.size == 1 or self.time_points == 1:
                        self.u0_loc = self.u_last_loc.copy()

                    # spatial parallelization
                    elif self.frac > 1:
                        self.u0_loc = self.comm_subcol_alternating.bcast(self.u_last_loc, root=self.size_subcol_alternating - 1)
                        if self.rank_subcol_alternating == self.size_subcol_alternating - 1:
                            self.u0_loc = self.u_last_loc

                    # without spatial but time_points > size_col
                    else:
                        self.u0_loc = self.comm_col.bcast(self.u_last_loc, root=self.size_col - 1)
                        if self.rank_col == self.size_col - 1:
                            self.u0_loc = self.u_last_loc

        max_time = MPI.Wtime() - time_beg
        self.algorithm_time = self.comm.allreduce(max_time, op=MPI.MAX)

        comm_time = self.communication_time
        self.communication_time = self.comm.allreduce(comm_time, op=MPI.MAX)
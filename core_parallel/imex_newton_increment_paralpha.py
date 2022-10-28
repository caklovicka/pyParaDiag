import numpy as np
from mpi4py import MPI
from core_parallel.helpers import Helpers
import os


class IMEXNewtonIncrementParalpha(Helpers):

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
            i_beta = -1
            t_start = self.T_start + self.time_intervals * rolling_interval * self.dt
            h0 = np.zeros(self.rows_loc, dtype=complex, order='C')  # initial guess for inner systems

            v_loc = self.__get_v__(t_start)     # the rhs of the all-at-once system

            if self.betas == NotImplemented:
                self.betas = [0]
            J_loc = None

            while self.iterations[rolling_interval] < self.maxiter and not self.stop:       # main iterations

                i_alpha = self.__next_alpha__(i_alpha)
                i_beta = self.__next_beta__(i_beta)

                res_loc = self.__get_linear_residual__(v_loc)       # rhs vector of the iteration
                res_loc += self.__get_F_residual__()                # add the explicit part
                res_norm = self.__get_max_norm__(res_loc)

                self.residual[rolling_interval].append(res_norm)
                if self.residual[rolling_interval][-1] <= self.tol:
                    break

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
                h1_loc = self.__solve_substitution__(Cinv, h_loc)       # step 4 ... (C x I) h1 = h

                self.system_time_max[rolling_interval].append(self.comm.allreduce(max(system_time), op=MPI.MAX))
                self.system_time_min[rolling_interval].append(self.comm.allreduce(min(system_time), op=MPI.MIN))
                self.solver_its_max[rolling_interval].append(self.comm.allreduce(max(its), op=MPI.MAX))
                self.solver_its_min[rolling_interval].append(self.comm.allreduce(min(its), op=MPI.MIN))
                self.inner_tols.append(self.stol)

                h_loc = self.__get_ifft_h__(h1_loc, self.alphas[i_alpha])       # solving (Sinv x I) h1_loc = h with ifft

                # ------ PROCESSORS HAVE NORMAL INDICES ROM HERE! -------

                self.iterations[rolling_interval] += 1
                self.u_loc += h_loc     # update the solution
                self.consecutive_error[rolling_interval].append(self.__get_max_norm__(h_loc))   # consecutive error, error of the increment

                if self.iterations[rolling_interval] == self.maxiter:
                    self.stop = True

                #self.__print_on_runtime__(t_start, rolling_interval)

                # end of main iterations (while loop)

            self.__write_u_in_txt__(rolling_interval)   # document writing

            if rolling_interval + 1 < self.rolling:     # update u0_loc (new initial condition) on processors that need it (first column) if we are not in the last rolling interval
                self.__fill_u_last__(fill_old=False)
                self.__bcast_u_last_loc__()
                if self.comm_last != MPI.COMM_NULL:
                    self.u0_loc = self.u_last_loc.copy()

        max_time = MPI.Wtime() - time_beg
        self.algorithm_time = self.comm.allreduce(max_time, op=MPI.MAX)

        comm_time = self.communication_time
        self.communication_time = self.comm.allreduce(comm_time, op=MPI.MAX)
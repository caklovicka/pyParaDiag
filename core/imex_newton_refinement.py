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

    def solve(self):

        self.comm.Barrier()
        time_beg = MPI.Wtime()

        h0 = np.zeros(self.rows_loc, dtype=complex, order='C')  # initial guess for inner systems
        self.stop = False
        self.stop_outer = False

        self.__fill_initial_guesses__()
        if self.time_points == 1:
            v_loc = self.__get_v_Euler__()
        else:
            # TODO: implement
            raise RuntimeError('Not implemented for M > 1')

        while not self.stop_outer:                      # main iterations
            while not self.stop:                        # paradiag iters

                # compute residual
                if self.time_points == 1:
                    res_loc = self.__get_linear_residual_Euler__(v_loc)
                else:
                    # TODO: implement
                    raise RuntimeError('Not implemented for M > 1')

                res_norm = self.__get_max_norm__(res_loc)
                self.residual.append(res_norm)

                # if it did not converge for a given maximum iterations
                if self.iterations == self.paradiag_maxiter and self.residual[-1] > self.paradiag_tol:
                    self.convergence = 0
                    break

                # if the solution starts exploding, terminate earlier
                if self.residual[-1] > 1000:
                    self.convergence = 0
                    print('divergence? residual = ', self.residual[-1])
                    break

                # do a parallel scaled FFT in time
                g_loc, Rev = self.__get_fft__(res_loc, self.alpha)
                print(self.rank_global, int(Rev, 2), g_loc.real)
                exit()

                # ------ PROCESSORS HAVE DIFFERENT INDICES ROM HERE! -------

                system_time = []
                its = []

                # get shifted systems
                if self.time_points == 1:
                    d = self.__get_shift_Euler__(int(Rev, 2), self.alpha)
                else:
                    # TODO: implement
                    raise RuntimeError('Not implemented for M > 1')

                # solve inner systems
                if self.time_points == 1:
                    time_solver = MPI.Wtime()
                    h1_loc, it = self.__solve_shifted_systems_Euler__(g_loc, d, h0.copy(), self.solver_tol)
                    system_time.append(MPI.Wtime() - time_solver)
                else:
                    # TODO: implement all the substeps
                    raise RuntimeError('Not implemented for M > 1')

                its.append(it)

                self.system_time_max.append(self.comm.allreduce(max(system_time), op=MPI.MAX))
                self.system_time_min.append(self.comm.allreduce(min(system_time), op=MPI.MIN))
                self.solver_its_max.append(self.comm.allreduce(max(its), op=MPI.MAX))
                self.solver_its_min.append(self.comm.allreduce(min(its), op=MPI.MIN))

                # do an ifft
                h_loc = self.__get_ifft_h__(h1_loc, self.alpha)

                # ------ PROCESSORS HAVE NORMAL INDICES ROM HERE! -------

                if self.state:
                    self.y_loc += h_loc
                elif self.adjoint:
                    self.p_loc += h_loc

                self.iterations += 1

            # compute gradient on the state, None on the adjoint
            grad_loc = self.__get_gradient__()

            if self.state:
                # evaluate objective on state
                obj = self.__get_objective__()

                # update errors
                grad_loc_scaled = np.sqrt(self.dt * np.prod(self.dx)) * np.linalg.norm(grad_loc, 2)
                # TODO: get grad_max_grad_scaled and that is the stopping criterion for stop_outer
                self.grad_err.append(grad_loc_scaled)
                self.obj_err.append(obj)

                # update u
                self.__get_u__(grad_loc)

            self.outer_iterations += 1
            if self.outer_iterations >= self.outer_maxiter:
                self.stop_outer = True
            # end of main iterations (while loop)

        max_time = MPI.Wtime() - time_beg
        self.algorithm_time = self.comm.allreduce(max_time, op=MPI.MAX)

        comm_time = self.communication_time
        self.communication_time = self.comm.allreduce(comm_time, op=MPI.MAX)

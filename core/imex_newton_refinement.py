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

        self.stop_outer = False

        self.__fill_initial_guesses__()
        if self.collocation_points == 1:
            v_loc = self.__get_v_Euler__()
        else:
            # TODO: implement
            raise RuntimeError('Not implemented for M > 1')

        while not self.stop_outer:                      # outer iterations
            self.residual.append([])
            self.paradiag_iterations.append(0)

            # do paradiag
            self.__paradiag__(v_loc)

            # compute gradient on the state, None on the adjoint
            grad_loc = self.__get_gradient__()
            grad_norm_scaled = self.__get_grad_norm_scaled__(grad_loc)

            # evaluate objective on state
            obj = self.__get_objective__()

            # update errors
            if self.state:
                self.grad_err.append(grad_norm_scaled)
                self.obj_err.append(obj)

            # check if we can terminate here before doing the u_try
            if grad_norm_scaled <= self.outer_tol:
                if self.state:
                    self.u -= self.step * grad_loc
                break

            # update u with multiple tries
            self.__get_u__(grad_loc, v_loc)

            self.outer_iterations += 1
            if self.outer_iterations >= self.outer_maxiter:
                self.stop_outer = True
            # end of main iterations (while loop)

        max_time = MPI.Wtime() - time_beg
        self.algorithm_time = self.comm.allreduce(max_time, op=MPI.MAX)

        comm_time = self.communication_time
        self.communication_time = self.comm.allreduce(comm_time, op=MPI.MAX)

import numpy as np
from mpi4py import MPI
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from core_parallel.linear_helpers import LinearHelpers
import os
from scipy import sparse

np.set_printoptions(precision=5, linewidth=np.inf)


class LinearParalpha(LinearHelpers):

    def __init__(self):

        LinearHelpers.__init__(self)
        self.setup_var = False

    def setup(self):

        self.setup_var = True

        super(LinearHelpers, self).setup()

        assert self.proc_col > 0, 'proc_col = {} should be > 0'.format(self.proc_col)
        assert self.proc_row > 0, 'proc_row = {} should be > 0'.format(self.proc_row)
        assert np.log2(self.time_intervals) - int(np.log2(self.time_intervals)) < 0.1, 'time_intervals = {} should be power of 2.'.format(self.time_intervals)
        assert self.proc_col * self.proc_row == MPI.COMM_WORLD.Get_size(), 'Please input a sufficient amount of processors. You need {} and you have proc_col * proc_row = {}'.format(self.proc_col * self.proc_row, self.size)

        assert self.time_intervals == self.proc_row, 'time_intervals = {} has to be equal to proc_row = {}.'.format(
            self.time_intervals, self.proc_row)

        if self.proc_col >= self.time_points:
            assert self.proc_col % self.time_points == 0, 'proc_col = {} has to be divisible by time_points = {}'.format(
                self.proc_col, self.time_points)
            assert self.global_size_A * self.time_points % self.proc_col == 0, 'dimA * self.time_points = {} should be divisible by proc_col = {}'.format(
                self.global_size_A * self.time_points, self.proc_col)
            assert self.proc_col <= self.global_size_A * self.time_points, 'proc_col = {} has to be less or equal to (dimA * time_points) = {}'.format(
                self.proc_col, self.global_size_A * self.time_points)
            assert self.proc_col >= self.time_points, 'proc_col = {} should be at least as time_points = {}'.format(
                self.proc_col, self.time_points)
        else:
            assert self.time_points % self.proc_col == 0, 'time_points = {} should be divisible by proc_col = {}'.format(
                self.time_points, self.proc_col)

        if self.time_intervals > 1 and self.optimal_alphas is False:
            assert len(self.alphas) >= 1, 'Please define a list of alphas, or put optimal_alphas=True'

        if self.optimal_alphas is True:
            self.alphas = list()
            if self.time_points == 2:
                self.bad_alphas = np.array([0.323, 0.477]) ** self.time_intervals
            elif self.time_points == 3:
                self.bad_alphas = np.array([0.516, 0.504, 0.069]) ** self.time_intervals
        if self.time_intervals == 1:
            self.optimal_alphas = False
            self.alphas = [0]
            self.maxiter = 1

        # build variables
        self.dt = (self.T_end - self.T_start) / (self.time_intervals * self.rolling)
        coll = CollGaussRadau_Right(num_nodes=self.time_points, tleft=0, tright=1)
        self.t = self.dt * np.array(coll.nodes)

        # case with spatial parallelization
        if self.frac > 1:
            self.u0_loc = self.u_initial(self.x).flatten()[self.rank_subcol_seq * self.rows_loc: (self.rank_subcol_seq + 1) * self.rows_loc]
        # case without spatial parallelization
        else:
            self.u0_loc = self.u_initial(self.x).flatten()

        self.u_last_loc = self.u0_loc.copy(order='C')
        self.u_loc = np.empty((self.rows_loc, self.cols_loc), dtype=complex, order='C')
        self.u_last_old_loc = None

        self.algorithm_time = 0
        self.communication_time = 0
        self.system_time_max = []
        self.system_time_min = []
        self.inner_tols = []

        self.viable = 'Viability of the problem is not known.'
        self.err_last = list()
        self.iterations = np.zeros(self.rolling)

        # build matrices and vectors
        self.Q = coll.Qmat[1:, 1:]

        self.P = np.zeros((self.time_points, self.time_points))
        for i in range(self.time_points):
            self.P[i, -1] = 1
        self.P = sparse.csr_matrix(self.P)

        # documents
        if self.rank == 0 and self.document is not 'None':
            self.time_document = self.document + '_times'
            if os.path.exists(self.document):
                os.remove(self.document)
            if os.path.exists(self.time_document):
                os.remove(self.time_document)
            file = open(self.document, "w+")
            file.close()
            self.__write_time_in_txt__()

    def solve(self):
        self.comm.Barrier()
        time_beg = MPI.Wtime()

        h_loc = np.empty((self.rows_loc, self.cols_loc), dtype=complex, order='C')
        # h1_loc_old = np.zeros((self.rows_loc, self.cols_loc), dtype=complex, order='C')
        h1_loc = np.empty((self.rows_loc, self.cols_loc), dtype=complex, order='C')

        for rolling_interval in range(self.rolling):

            self.err_last.append([])
            self.err_last[rolling_interval].append(np.inf)
            self.system_time_max.append([])
            self.system_time_min.append([])
            i_alpha = -1
            t_start = self.T_start + self.time_intervals * rolling_interval * self.dt

            # build local v
            v_loc = self.__get_v__(t_start)
            self.comm.Barrier()

            # save v1 on the processors that have v1
            if self.rank_row == 0:
                v1_loc = v_loc[:, 0].copy()
            else:
                v1_loc = None

            r = None
            m0 = self.m0
            eps = np.finfo(complex).eps
            gamma = self.time_intervals * (3 * eps + self.stol)
            if self.optimal_alphas is True:
                r = self.__get_r__(v_loc)
                self.comm.Barrier()
                if self.rank == 0:
                    print('m0 = ', m0, 'r = ', r, flush=True)
            self.stop = False

            # main iterations
            while self.iterations[rolling_interval] < self.maxiter and not self.stop:

                self.iterations[rolling_interval] += 1

                if self.optimal_alphas is True:
                    self.alphas.append(np.sqrt((gamma * r)/m0))
                    m0 = 2 * np.sqrt(gamma * m0 * r)

                    # evasion = False
                    #
                    # # in case we have to evade an alpha
                    # if self.time_points > 1:
                    #     for ba in self.bad_alphas:
                    #         if 1.0 / 3 < abs(self.alphas[-1] / ba) < 3:
                    #             self.alphas[-1] = min(5 * self.alphas[-1], 0.1)
                    #             m0 = self.alphas[-1] * m0 + gamma * r / self.alphas[-1]
                    #             if self.rank == 0:
                    #                 print('NEW m0 = ', m0, 'alpha = ', self.alphas[-1])
                    #             evasion = True
                    #             break
                    #
                    #     if evasion is False:
                    #         m0 = 2 * np.sqrt(gamma * m0 * r)
                    #
                    # else:
                    #     m0 = 2 * np.sqrt(gamma * m0 * r)

                    if self.rank == 0:
                        print('m = ', m0, 'alpha = ', self.alphas[-1], 'err_max = ', self.err_last[rolling_interval][-1], flush=True)
                    if m0 <= self.tol:
                        self.stop = True

                i_alpha = self.__next_alpha__(i_alpha)

                # assemble the residual vector
                w_loc = self.__get_w__(self.alphas[i_alpha], v_loc, v1_loc)

                # solving (S x I) g = w with ifft
                g_loc, Rev = self.__get_fft__(w_loc, self.alphas[i_alpha])

                # ------ PROCESSORS HAVE DIFFERENT INDICES ROM HERE! -------

                # solve local systems in 4 steps with diagonalization of QCinv
                system_time = []
                for k in range(self.cols_loc):
                    l_new = int(Rev[k], 2)
                    Dl_new = -self.alphas[i_alpha] ** (1 / self.time_intervals) * np.exp(-2 * np.pi * 1j * l_new / self.time_intervals)
                    C = Dl_new * self.P + np.eye(self.time_points)  # same for every proc in the same column

                    Cinv = np.linalg.inv(C)
                    R = self.Q @ Cinv
                    D, Z = np.linalg.eig(R)
                    Zinv = np.linalg.inv(Z)     # Z @ D @ Zinv = R

                    # step 1 ... (Z x I) h = g
                    h_loc[:, k] = self.__step1__(Zinv, g_loc[:, k])

                    # step 2 ... solve local systems (I - Di * A) h1 = h
                    time_solver = MPI.Wtime()
                    h0 = np.zeros((self.rows_loc, self.cols_loc), dtype=complex, order='C')
                    h1_loc[:, k], it = self.__step2__(h_loc[:, k], D, h0, self.stol)
                    system_time.append(MPI.Wtime() - time_solver)
                    print('on proc ', self.rank, ', iterations = ', it)
                    #h1_loc_old[:, k] = h1_loc[:, k] #if this is uncommented, then the initial guess is not zeros

                    # step 3 ... (Zinv x I) h = h1
                    h_loc[:, k] = self.__step1__(Z, h1_loc[:, k])

                    # step 4 ... (C x I) h1 = h
                    self.u_loc[:, k] = self.__step1__(Cinv, h_loc[:, k])

                self.system_time_max[rolling_interval].append(self.comm.allreduce(max(system_time), op=MPI.MAX))
                self.system_time_min[rolling_interval].append(self.comm.allreduce(min(system_time), op=MPI.MIN))

                self.inner_tols.append(self.stol)

                # solving (Sinv x I) h1_loc = u with fft
                self.__get_ifft__(self.alphas[i_alpha])

                # ------ PROCESSORS HAVE NORMAL INDICES ROM HERE! -------

                # the processors that contain u_last have to decide whether to finish and compute the whole u or move on
                # broadcast the error, a stopping criteria
                # updates u_last_loc and u_last_loc_old
                err_max = self.__get_u_last__()
                self.err_last[rolling_interval].append(err_max)

                # update u_last_loc on processors that need it (first column) if we are moving on
                if self.err_last[rolling_interval][-1] < self.tol or self.iterations[rolling_interval] == self.maxiter:
                    self.stop = True

                #if (1 < self.rolling != rolling_interval + 1) or not self.stop:
                self.__bcast_u_last_loc__()

                # DELETE
                if self.rank == self.size - 1:#self.size_subcol_seq:
                    exact = self.u_exact(t_start + self.dt * self.time_intervals, self.x).flatten()[self.row_beg:self.row_end]
                    approx = self.u_last_loc.flatten()
                    d = exact - approx
                    d = d.flatten()
                    err_abs = np.linalg.norm(d, np.inf)
                    err_rel = np.linalg.norm(d, np.inf) / np.linalg.norm(exact, np.inf)
                    print('on {},  abs, rel err inf norm [from paralpha] = {}, {}, iter = {}, rolling = {}'.format(self.rank, err_abs, err_rel, int(self.iterations[rolling_interval]), rolling_interval), flush=True)
                # DELETE

                # end of main iterations (while loop)

            # document writing
            if self.document is not 'None':
                self.__write_u_in_txt__(rolling_interval)
                self.comm.Barrier()

            # update u0_loc (new initial condition) on processors that need it (first column) if we are not in the last rolling interval
            if rolling_interval + 1 < self.rolling:

                if self.comm_last is not 'None' and self.time_intervals > 1:
                    self.u0_loc = self.u_last_loc.copy()

                # to support a sequrntial run
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

    def summary(self, details=False):

        if self.rank == 0:
            assert self.setup_var is True, 'Please call the setup function before summary.'
            print('-----------------------< summary >-----------------------')
            print('solving on T1 = {}, T2 = {}'.format(self.T_start, self.T_end), flush=True)
            print('no. of spatial points = {}'.format(self.spatial_points), flush=True)
            print('dx = {}'.format(self.dx), flush=True)
            print('no. of time points on an interval = {}'.format(self.time_points), flush=True)
            if details:
                print('    {}'.format(self.t), flush=True)
            print('no. of time intervals = {}'.format(self.time_intervals), flush=True)
            print('no. of alphas = {}'.format(len(self.alphas)), flush=True)
            if details:
                print('    {}'.format(self.alphas), flush=True)
            print(self.viable, flush=True)
            print('rolling intervals = {}'.format(self.rolling), flush=True)
            print('dt (of one SDC interval) = {}'.format(self.dt), flush=True)
            print('processors for spatial parallelization for solving (I - Q x A) are {}'.format(self.proc_col), flush=True)
            print('processors for time interval parallelization are {}'.format(self.proc_row), flush=True)
            print('maxiter = {}'.format(self.maxiter), flush=True)
            print('output document = {}'.format(self.document), flush=True)
            print('tol = {}'.format(self.tol), flush=True)
            print('last error = {}'.format(self.err_last), flush=True)
            print('iterations of Paralpha = {}'.format(self.iterations), flush=True)
            print('max iterations of Paralpha = {}'.format(max(self.iterations)), flush=True)
            print('algorithm time = {:.5f} s'.format(self.algorithm_time), flush=True)
            print('communication time = {:.5f} s'.format(self.communication_time), flush=True)
            print('inner solver = {}'.format(self.solver), flush=True)
            print('system_time_max = {}'.format(max(self.system_time_max)), flush=True)
            print('system_time_min = {}'.format(min(self.system_time_min)), flush=True)
            print('inner solver tols = {}'.format(self.inner_tols), flush=True)
            print('inner solver maxiter = {}'.format(self.smaxiter), flush=True)
            print('-----------------------< end summary >-----------------------')
import numpy as np
from core.communicators import Communicators
from pySDC.core.Collocation import CollBase
from scipy import sparse
from mpi4py import MPI
from petsc4py import PETSc
import scipy as sc
from scipy.sparse import linalg
import os


class Helpers(Communicators):

    def __init__(self):
        super().__init__()

    def setup(self):

        super().setup()

        # assertions
        assert self.proc_col > 0, 'proc_col = {} should be > 0'.format(self.proc_col)
        assert self.proc_row > 0, 'proc_row = {} should be > 0'.format(self.proc_row)
        assert np.log2(self.time_intervals) - int(np.log2(self.time_intervals)) < 0.1, \
            'time_intervals = {} should be power of 2.'.format(self.time_intervals)
        assert 2 * self.proc_col * self.proc_row == MPI.COMM_WORLD.Get_size(), \
            'Please input a sufficient amount of processors. You need {} and you have 2 * proc_col * proc_row = {}'\
            .format(self.proc_col * self.proc_row, self.size)
        assert self.time_intervals == self.proc_row, 'time_intervals = {} has to be equal to proc_row = {}.'\
            .format(self.time_intervals, self.proc_row)

        if self.proc_col >= self.time_points:
            assert self.proc_col % self.time_points == 0, 'proc_col = {} has to be divisible by time_points = {}'\
                .format(self.proc_col, self.time_points)
            assert self.global_size_A * self.time_points % self.proc_col == 0, \
                'dimA * self.time_points = {} should be divisible by proc_col = {}'\
                .format(self.global_size_A * self.time_points, self.proc_col)
            assert self.proc_col <= self.global_size_A * self.time_points, \
                'proc_col = {} has to be less or equal to (dimA * time_points) = {}'\
                .format(self.proc_col, self.global_size_A * self.time_points)
            assert self.proc_col >= self.time_points, 'proc_col = {} should be at least as time_points = {}'\
                .format(self.proc_col, self.time_points)
        else:
            assert self.time_points % self.proc_col == 0, 'time_points = {} should be divisible by proc_col = {}'\
                .format(self.time_points, self.proc_col)

        # build variables
        self.dt = (self.T_end - self.T_start) / self.time_intervals
        coll = CollBase(self.time_points, 0, 1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')
        self.t = self.dt * np.array(coll.nodes)

        # fill initial conditions
        # case with spatial parallelization
        if self.frac > 1:
            self.y0_loc = self.y_initial(self.x).flatten()[self.row_beg: self.row_end]
            if self.adjoint:
                self.pT_loc = self.p_end(self.x).flatten()[self.row_beg: self.row_end]
        # case without spatial parallelization
        else:
            self.y0_loc = self.y_initial(self.x).flatten()
            if self.adjoint:
                self.pT_loc = self.p_end(self.x).flatten()

        if self.state:
            self.u_loc = np.zeros_like(self.y0_loc)

        # auxiliaries
        self.algorithm_time = 0
        self.communication_time = 0
        self.system_time_max = []
        self.system_time_min = []
        self.solver_its_max = []
        self.solver_its_min = []
        self.iterations = 0

        # build matrices for the collocation problem
        self.Q = coll.Qmat[1:, 1:]
        self.P = np.zeros((self.time_points, self.time_points))
        for i in range(self.time_points):
            self.P[i, -1] = 1
        self.P = sparse.csr_matrix(self.P)

    def gradient(self):
        # TODO
        return

    def objective(self):
        # TODO
        return

    def __fill_initial_guesses__(self):

        # case with spatial parallelization
        if self.frac > 1:
            if self.state:
                self.y_loc = self.y0_loc.copy(order='C').astype(complex)
            elif self.adjoint:
                self.p_loc = self.pT_loc.copy(order='C').astype(complex)

        # case without spatial parallelization
        else:
            if self.state:
                self.y_loc = np.tile(self.y0_loc, self.Frac).astype(complex)
            elif self.adjoint:
                self.p_loc = np.tile(self.pT_loc, self.Frac).astype(complex)

    def __get_v_Euler__(self):
        # on state: (y0, 0, ..., 0)
        # on adjoint: dt (yd_0, ..., yd_(L-1)) + (-dt y0, 0, ..., 0, pL)

        v = np.zeros(self.rows_loc, dtype=complex)

        if self.state:
            if self.rank_row == 0:
                v = self.y0_loc.copy(order='C').astype(complex)

        elif self.adjoint:
            v += self.dt * self.yd(self.rank_row * self.dt, self.x).flatten()[self.row_beg:self.row_end]

            if self.rank_row == 0:
                v -= self.dt * self.y0_loc

            elif self.rank_row == self.size_row - 1:
                v += self.pT_loc

        return v

    def __get_r__(self, v_loc):

        r = 0
        temp = 0
        if self.rank_row == 0:
            # with spatial parallelization
            if self.frac != 0:
                temp = np.linalg.norm(v_loc + self.u0_loc, np.infty)
            # without spatial parallelization
            else:
                for i in range(self.Frac):
                    temp = max(temp, np.linalg.norm(v_loc[i * self.global_size_A:(i+1) * self.global_size_A] + self.u0_loc, np.infty))
        else:
            temp = np.linalg.norm(v_loc, np.infty)
        r = max(r, temp)

        if self.size > 1:
            time_beg = MPI.Wtime()
            temp = self.comm.allreduce(r, op=MPI.MAX)
            self.communication_time += MPI.Wtime() - time_beg
            return temp

        else:
            return r

    # fft
    def __get_fft__(self, w_loc, a):

        if self.time_intervals == 1:
            return w_loc, '0'

        g_loc = a ** (self.rank_row / self.time_intervals) / self.time_intervals * w_loc    # scale
        n = int(np.log2(self.time_intervals))
        P = format(self.rank_row, 'b').zfill(n)  # binary of the rank in string
        R = P[::-1]  # reversed binary in string, index that the proc will have after ifft
        we = np.exp(-2 * np.pi * 1j / self.time_intervals)

        # stages of butterfly
        for k in range(n):
            p = self.time_intervals // 2 ** (k + 1)
            r = int(R, 2) % 2 ** (k + 1) - 2 ** k
            scalar = we ** (r * p)
            factor = 1

            if P[k] == '1':
                factor = -1
                if scalar != 1:  # multiply if the factor is != 1
                    g_loc *= scalar

            # make a new string and an int from it, a proc to communicate with
            comm_with = list(P)
            if comm_with[k] == '1':
                comm_with[k] = '0'
            else:
                comm_with[k] = '1'
            comm_with = int(''.join(comm_with), 2)

            # now communicate
            time_beg = MPI.Wtime()
            req = self.comm_row.isend(g_loc, dest=comm_with, tag=k)
            gr = self.comm_row.recv(source=comm_with, tag=k)
            req.Wait()
            self.communication_time += MPI.Wtime() - time_beg

            # glue the info
            g_loc = gr + factor * g_loc

        return g_loc, R

    def __get_w__(self, a, v_loc, v1=None):

        w_loc = v_loc.copy()
        if v1 is not None:

            # with spatial parallelization
            if self.frac > 1:
                w_loc = v1 + self.u0_loc - a * self.u_last_loc

            # without spatial parallelization
            else:
                for i in range(self.Frac):
                    w_loc[i * self.global_size_A:(i+1) * self.global_size_A] = self.u0_loc - a * self.u_last_loc
                w_loc += v1
        return w_loc

    def __get_Au__(self):

        Au_loc = np.zeros_like(self.u_loc)
        # case with spatial parallelization
        if self.frac > 1:
            A_petsc = PETSc.Mat()
            csr = (self.Apar.indptr, self.Apar.indices, self.Apar.data)
            A_petsc.createAIJWithArrays(size=(self.global_size_A, self.global_size_A), csr=csr, comm=self.comm_matrix)

            u_petsc = PETSc.Vec()
            u_petsc.createWithArray(array=self.u_loc, comm=self.comm_matrix)

            Au_petsc = PETSc.Vec()
            Au_petsc.createWithArray(Au_loc, comm=self.comm_matrix)

            A_petsc.mult(u_petsc, Au_petsc)
            Au_loc = Au_petsc.getArray()

            A_petsc.destroy()
            u_petsc.destroy()
            Au_petsc.destroy()

        # case without spatial parallelization
        else:
            for i in range(self.Frac):
                Au_loc[i * self.global_size_A: (i + 1) * self.global_size_A] = self.Apar @ self.u_loc[i * self.global_size_A: (i + 1) * self.global_size_A]

        return Au_loc

    def __get_Hu__(self):

        if self.time_intervals == 1:
            return self.u0_loc

        Hu_loc = None
        req = None

        # a horizontal send to the right
        # processors who send
        if self.rank_row < self.size_row - 1:

            # case with spatial parallelization
            if self.frac > 1:
                # just the last level
                if self.size_col - self.frac <= self.rank_col:
                    time_beg = MPI.Wtime()
                    req = self.comm_row.isend(self.u_loc, dest=self.rank_row + 1, tag=0)
                    self.communication_time += MPI.Wtime() - time_beg

            # case without spatial parallelization, just chunks of last
            else:
                # just the last level
                if self.rank_col == self.size_col - 1:
                    time_beg = MPI.Wtime()
                    req = self.comm_row.isend(self.u_loc[-self.global_size_A:], dest=self.rank_row + 1, tag=0)
                    self.communication_time += MPI.Wtime() - time_beg

        # processors who receive
        if self.rank_row > 0:

            # case with spatial parallelization
            if self.frac > 1:
                # just the last level
                if self.size_col - self.frac <= self.rank_col:
                    time_beg = MPI.Wtime()
                    Hu_loc = self.comm_row.recv(source=self.rank_row - 1, tag=0)
                    self.communication_time += MPI.Wtime() - time_beg

            # case without spatial parallelization
            else:
                # just the last level
                if self.rank_col == self.size_col - 1:
                    time_beg = MPI.Wtime()
                    Hu_loc = self.comm_row.recv(source=self.rank_row - 1, tag=0)
                    self.communication_time += MPI.Wtime() - time_beg

        # be sure to s
        time_beg = MPI.Wtime()
        if req is not None:
            req.Wait()
        self.communication_time += MPI.Wtime() - time_beg

        # a vertical broadcast
        # case with spatial parallelization
        if self.frac > 1:
            time_beg = MPI.Wtime()
            Hu_loc = self.comm_subcol_alternating.bcast(Hu_loc, root=self.size_subcol_alternating - 1)
            self.communication_time += MPI.Wtime() - time_beg

        # case without spatial parallelization
        elif self.size_col > 1:
            time_beg = MPI.Wtime()
            Hu_loc = self.comm_col.bcast(Hu_loc, root=self.size_col - 1)
            self.communication_time += MPI.Wtime() - time_beg

        if self.rank_row == 0:
            Hu_loc = self.u0_loc.copy()

        return Hu_loc

    def __get_linear_residual__(self, v_loc):

        Hu_loc = self.__get_Hu__()

        # case with spatial parallelization
        if self.frac > 0:
            res_loc = v_loc + Hu_loc
        # case without spatial parallelization
        else:
            res_loc = v_loc + np.tile(Hu_loc, self.Frac)

        Cu_loc = self.u_loc - self.dt * self.__solve_substitution__(self.Q, self.__get_Au__())
        res_loc -= Cu_loc

        return res_loc

    def __get_shifted_matrices__(self, l_new, a):

        Dl_new = -a ** (1 / self.time_intervals) * np.exp(-2 * np.pi * 1j * l_new / self.time_intervals)
        C = Dl_new * self.P + np.eye(self.time_points)  # same for every proc in the same column

        Cinv = np.linalg.inv(C)
        R = self.Q @ Cinv
        D, Z = np.linalg.eig(R)
        Zinv = np.linalg.inv(Z)  # Z @ D @ Zinv = R

        return Zinv, D, Z, Cinv

    def __get_max_norm__(self, c):

        err_loc = self.norm(c)

        if self.size == 1:
            return err_loc

        time_beg = MPI.Wtime()
        err_max = self.comm.allreduce(err_loc, op=MPI.MAX)
        self.communication_time += MPI.Wtime() - time_beg

        return err_max

    def __solve_substitution__(self, Zinv, g_loc):

        h_loc = np.empty_like(g_loc, dtype=complex)

        # case with spatial parallelization
        if self.frac > 1:
            for proc in range(self.size_subcol_alternating):
                h_scaled = Zinv[proc, self.rank_subcol_alternating] * g_loc

                time_beg = MPI.Wtime()
                temp = self.comm_subcol_alternating.reduce(h_scaled, op=MPI.SUM, root=proc)
                self.communication_time += MPI.Wtime() - time_beg

                if proc == self.rank_subcol_alternating:
                    h_loc = temp.copy(order='C')

        # case without spatial parallelization
        else:
            for proc in range(self.proc_col):
                h_scaled = np.zeros_like(g_loc, dtype=complex, order='C')
                for i in range(self.Frac):
                    for j in range(self.Frac):
                        h_scaled[i*self.global_size_A:(i+1)*self.global_size_A] += Zinv[i + proc * self.Frac, j + self.rank_col * self.Frac] * g_loc[j*self.global_size_A:(j+1)*self.global_size_A]

                if self.size_col > 1:
                    time_beg = MPI.Wtime()
                    temp = self.comm_col.reduce(h_scaled, op=MPI.SUM, root=proc)
                    self.communication_time += MPI.Wtime() - time_beg

                    if proc == self.rank_col:
                        h_loc = temp.copy(order='C')

                else:
                    return h_scaled

        # self.comm.Barrier()
        return h_loc

    def __solve_inner_systems__(self, h_loc, D, x0, tol):
        it_max = 0

        # case with spatial parallelization
        if self.row_end - self.row_beg != self.global_size_A:
            sys = sc.sparse.eye(m=self.row_end - self.row_beg, n=self.global_size_A, k=self.row_beg) - self.dt * D[self.rank_subcol_alternating] * self.Apar
            h1_loc, it = self.linear_solver(sys, h_loc, x0, tol)
            it_max = max(it, it_max)

        # case without spatial parallelization
        else:
            h1_loc = np.zeros_like(h_loc, dtype=complex, order='C')
            for i in range(self.Frac):
                sys = sc.sparse.eye(self.global_size_A) - self.dt * D[i + self.rank_col * self.Frac] * self.Apar
                if self.solver == 'custom':
                    h1_loc[i * self.global_size_A:(i + 1) * self.global_size_A], it = self.linear_solver(sys, h_loc[i * self.global_size_A:(i + 1) * self.global_size_A], x0[i * self.global_size_A:(i + 1) * self.global_size_A], tol)
                    it_max = max(it, it_max)
                else:
                    h1_loc[i * self.global_size_A:(i + 1) * self.global_size_A], it = self.__linear_solver__(sys, h_loc[i * self.global_size_A:(i + 1) * self.global_size_A], x0[i * self.global_size_A:(i + 1) * self.global_size_A], tol)
                    it_max = max(it, it_max)

        return h1_loc, it_max

    def __get_ifft_h__(self, h1_loc, a):

        if self.time_intervals == 1:
            return h1_loc

        n = int(np.log2(self.time_intervals))
        P = format(self.rank_row, 'b').zfill(n)  # binary of the rank in string
        R = P[::-1]  # reversed binary in string, index that the proc will have after ifft
        we = np.exp(2 * np.pi * 1j / self.time_intervals)

        # stages of butterfly
        for k in range(n):
            p = self.time_intervals // 2 ** (n - k)
            r = int(R, 2) % 2 ** (n - k) - 2 ** (n - k - 1)
            scalar = we ** (r * p)
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
            time_beg = MPI.Wtime()
            req = self.comm_row.isend(h1_loc, dest=comm_with, tag=k)
            hr = self.comm_row.recv(source=comm_with, tag=k)
            req.Wait()
            self.communication_time += MPI.Wtime() - time_beg

            # glue the info
            h1_loc = hr + factor * h1_loc

            # scale the output
            if R[k] == '1' and scalar != 1:
                h1_loc *= scalar

        h1_loc *= a ** (-self.rank_row / self.time_intervals)

        return h1_loc

    def __bcast_u_last_loc__(self):

        if self.time_intervals == 1:
            return

        if self.comm_last != MPI.COMM_NULL:  # and self.size_col < self.size:
            time_beg = MPI.Wtime()
            self.u_last_loc = self.comm_last.bcast(self.u_last_loc, root=0)
            self.communication_time += MPI.Wtime() - time_beg

    # ifft
    def __get_ifft__(self, a):

        if self.time_intervals == 1:
            return

        n = int(np.log2(self.time_intervals))
        P = format(self.rank_row, 'b').zfill(n)  # binary of the rank in string
        R = P[::-1]  # reversed binary in string, index that the proc will have after ifft
        we = np.exp(2 * np.pi * 1j / self.time_intervals)

        # stages of butterfly
        for k in range(n):
            p = self.time_intervals // 2 ** (n - k)
            r = int(R, 2) % 2 ** (n - k) - 2 ** (n - k - 1)
            scalar = we ** (r * p)
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
            time_beg = MPI.Wtime()
            req = self.comm_row.isend(self.u_loc, dest=comm_with, tag=k)
            ur = self.comm_row.recv(source=comm_with, tag=k)
            req.Wait()
            self.communication_time += MPI.Wtime() - time_beg

            # glue the info
            self.u_loc = ur + factor * self.u_loc

            # scale the output
            if R[k] == '1' and scalar != 1:
                self.u_loc *= scalar

        self.u_loc *= a**(-self.rank_row / self.time_intervals)

    def __fill_u_last__(self, fill_old):

        # case with spatial parallelization, need reduction for maximal error
        if self.frac > 1:
            if self.size - self.size_subcol_seq <= self.rank:
                if fill_old:
                    self.u_last_old_loc = self.u_last_loc.copy()
                self.u_last_loc = self.u_loc.copy()

        # case without spatial parallelization, the whole vector is on the last processor
        else:
            if self.rank == self.size - 1:
                if fill_old:
                    self.u_last_old_loc = self.u_last_loc.copy()
                self.u_last_loc = self.u_loc[-self.global_size_A:]

    def __fill_u0_loc__(self):

        if self.size == 1:
            self.u0_loc = self.u_loc[-self.global_size_A:]

        elif self.time_intervals == 1:

            self.__fill_u_last__(fill_old=False)

            # with spatial parallelization
            if self.frac > 1:
                if self.rank_subcol_alternating == self.size_subcol_alternating - 1:
                    self.u0_loc = self.u_last_loc.copy()

                time_beg = MPI.Wtime()
                self.u0_loc = self.comm_subcol_alternating.bcast(self.u0_loc, root=self.size_subcol_alternating - 1)
                self.communication_time += MPI.Wtime() - time_beg

            # without spatial parallelization
            else:
                if self.rank_col == self.size_col - 1:
                    self.u0_loc = self.u_last_loc.copy()

                time_beg = MPI.Wtime()
                self.u0_loc = self.comm_col.bcast(self.u0_loc, root=self.size_col - 1)
                self.communication_time += MPI.Wtime() - time_beg

        else:
            self.__fill_u_last__(fill_old=False)
            self.__bcast_u_last_loc__()

            if self.rank_row == 0:
                self.u0_loc = self.u_last_loc.copy()

            time_beg = MPI.Wtime()
            self.u0_loc = self.comm_row.bcast(self.u0_loc, root=0)
            self.communication_time += MPI.Wtime() - time_beg

    def __get_consecutive_error_last__(self):

        err_max = 0

        # case with spatial parallelization, need reduction for maximal error
        if self.frac > 1:
            if self.size - self.size_subcol_seq <= self.rank:
                err_loc = self.norm(self.u_last_old_loc - self.u_last_loc)

                time_beg = MPI.Wtime()
                err_max = self.comm_subcol_seq.allreduce(err_loc, op=MPI.MAX)
                self.communication_time += MPI.Wtime() - time_beg

            # broadcast the error, a stopping criteria
            time_beg = MPI.Wtime()
            err_max = self.comm.bcast(err_max, root=self.size - 1)
            self.communication_time += MPI.Wtime() - time_beg

        # case without spatial parallelization, the whole vector is on the last processor
        else:
            if self.rank == self.size - 1:
                err_max = self.norm(self.u_last_old_loc - self.u_last_loc)

            # broadcast the error, a stopping criteria
            if self.size > 1:
                time_beg = MPI.Wtime()
                err_max = self.comm.bcast(err_max, root=self.size - 1)
                self.communication_time += MPI.Wtime() - time_beg

        return err_max

    # solver (space parallelization not included yet)
    def __linear_solver__(self, M_loc, m_loc, m0, tol):

        class gmres_counter(object):

            def __init__(self, disp=True):
                self._disp = disp
                self.niter = 0

            def __call__(self, rk=None):
                self.niter += 1

        counter = gmres_counter()
        it = 0

        if self.solver == 'gmres':
            x_loc, info = linalg.gmres(M_loc, m_loc, tol=tol, atol=0, maxiter=self.smaxiter, x0=m0, callback=counter)
            it = counter.niter

        else:
            x_loc = linalg.spsolve(M_loc, m_loc)

        return x_loc, it

    def __print_on_runtime__(self, t_start, rolling_interval):

        if self.rank == self.size - 1:  # self.size_subcol_seq:
            exact = self.u_exact(t_start + self.dt * self.time_intervals, self.x).flatten()[self.row_beg:self.row_end]
            approx = self.u_loc[-self.global_size_A:]
            d = exact - approx
            d = d.flatten()
            err_abs = np.linalg.norm(d, np.inf)
            err_rel = np.linalg.norm(d, np.inf) / np.linalg.norm(exact, np.inf)
            print('on {},  abs, rel err inf norm = {}, {}, iter = {}, rolling = {}'.format(self.rank, err_abs, err_rel, int(self.iterations[rolling_interval]), rolling_interval), flush=True)

    def summary(self, details=False):

        np.set_printoptions(precision=5, linewidth=np.inf)
        if self.rank == 0:
            assert self.setup_var is True, 'Please call the setup function before summary.'
            print('----------------')
            print(' discretization ')
            print('----------------')
            print('solving on [{}, {}]'.format(self.T_start, self.T_end), flush=True)
            print('no. of spatial points = {}, dx = {}'.format(self.spatial_points, self.dx), flush=True)
            print('no. of time intervals = {}, dt = {}'.format(self.time_intervals, self.dt), flush=True)
            print('collocation points = {}'.format(self.t), flush=True)
            print('alphas = {}'.format(self.alphas), flush=True)
            print('betas = {}'.format(self.betas), flush=True)
            print('rolling intervals = {}'.format(self.rolling), flush=True)

            print()
            print('-----------------')
            print(' parallelization ')
            print('-----------------')
            print('proc. for collocation-space problem: {}'.format(self.proc_col), flush=True)
            print('proc. for time problem: {}'.format(self.proc_row), flush=True)
            print('total: {}'.format(self.size))

            print()
            print('-------')
            print(' other ')
            print('-------')
            print('maxiter of paradiag = {}'.format(int(self.maxiter)), flush=True)
            print('output document = {}'.format(self.document), flush=True)
            print('tol = {}'.format(self.tol), flush=True)
            print('inner solver = {}'.format(self.solver), flush=True)
            print('inner solver tol = {}'.format(self.stol), flush=True)
            print('inner solver maxiter = {}'.format(self.smaxiter), flush=True)

            print()
            print('--------')
            print(' output ')
            print('--------')
            print('convergence = {}'.format(self.convergence))
            print('iterations of paradiag = {}'.format(self.iterations), flush=True)
            print()
            print('algorithm time = {:.5f} s'.format(self.algorithm_time), flush=True)
            print('communication time = {:.5f} s'.format(self.communication_time), flush=True)
            print()

            for i in range(self.rolling):
                if self.consecutive_err_last != NotImplemented:
                    self.consecutive_err_last[i] = [float("{:.5e}".format(elem)) for elem in self.consecutive_err_last[i]]
                if self.consecutive_error != NotImplemented:
                    self.consecutive_error[i] = [float("{:.5e}".format(elem)) for elem in self.consecutive_error[i]]
                if self.residual != NotImplemented:
                    self.residual[i] = [float("{:.5e}".format(elem)) for elem in self.residual[i]]

            if self.consecutive_err_last != NotImplemented:
                print('consecutive errors (last) = ', flush=True)
                for i in self.consecutive_err_last:
                    print(i, flush=True)

            if self.consecutive_error != NotImplemented:
                print('consecutive errors = ', flush=True)
                for i in self.consecutive_error:
                    print(i, flush=True)

            if self.residual != NotImplemented:
                print('residuals = ', flush=True)
                for i in self.residual:
                    print(i, flush=True)
            print()

            if details:
                for i in range(self.rolling):
                    self.system_time_max[i] = [float("{:.2e}".format(elem)) for elem in self.system_time_max[i]]
                    self.system_time_min[i] = [float("{:.2e}".format(elem)) for elem in self.system_time_min[i]]

                print('system_time_max =', flush=True)
                for i in self.system_time_max:
                    print(i, flush=True)
                print('system_time_min = ', flush=True)
                for i in self.system_time_min:
                    print(i, flush=True)
                print('solver_its_max =', flush=True)
                for i in self.solver_its_max:
                    print(i, flush=True)
                print('solver_its_min =', flush=True)
                for i in self.solver_its_min:
                    print(i, flush=True)
                #print('inner_tols =', flush=True)
                #for i in self.inner_tols:
                #    print(i, flush=True)
            print('-----------------------< end summary >-----------------------')
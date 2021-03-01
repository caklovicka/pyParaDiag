import numpy as np
from core_parallel.communicators import Communicators
from mpi4py import MPI
import scipy as sc


class LinearHelpers(Communicators):

    def __init__(self):
        Communicators.__init__(self)

    def __next_alpha__(self, idx):
        if idx + 1 < len(self.alphas) and self.time_intervals > 1:
            idx += 1
        return idx

    def __get_v__(self, t_start):
        v = np.zeros((self.rows_loc, self.cols_loc), dtype=complex)
        shift = self.rank_row * self.cols_loc

        # if we have spatial parallelization
        if self.frac > 1:
            for j in range(self.cols_loc):
                for k in range(self.time_points):
                    v[:, j] += self.dt * self.Q[self.rank_subcol_alternating, k] * self.bpar(t_start + self.t[k] + (shift + j) * self.dt)

        # case without spatial parallelization
        else:
            for i in range(self.Frac):
                for j in range(self.cols_loc):
                    for k in range(self.time_points):
                        v[i * self.global_size_A:(i+1)*self.global_size_A, j] += self.dt * self.Q[i + self.Frac * self.rank_col, k] * self.bpar(t_start + self.t[k] + (shift + j) * self.dt)
        return v

    def __get_r__(self, v_loc):

        r = 0
        temp = 0
        for j in range(self.cols_loc):
            if self.rank_row == 0:
                # with spatial parallelization
                if self.frac is not 0:
                    temp = np.linalg.norm(v_loc[:, j] + self.u0_loc, np.infty)
                # without spatial parallelization
                else:
                    for i in range(self.Frac):
                        temp = max(temp, np.linalg.norm(v_loc[i * self.global_size_A:(i+1) * self.global_size_A, j] + self.u0_loc, np.infty))
            else:
                temp = np.linalg.norm(v_loc[:, j], np.infty)
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
            return w_loc, ['0']

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

        return g_loc, [R]

    def __get_w__(self, a, v_loc, v1=None):

        w_loc = v_loc.copy()
        if v1 is not None:

            # with spatial parallelization
            if self.frac > 1:
                w_loc[:, 0] = v1 + self.u0_loc - a * self.u_last_loc

            # without spatial parallelization
            else:
                for i in range(self.Frac):
                    w_loc[i * self.global_size_A:(i+1) * self.global_size_A, 0] = self.u0_loc - a * self.u_last_loc
                w_loc[:, 0] += v1
        return w_loc

    def __step1__(self, Zinv, g_loc):

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

    def __step2__(self, h_loc, D, x0, tol):

        h1_loc = np.empty_like(h_loc, dtype=complex, order='C')
        # case with spatial parallelization
        if self.row_end - self.row_beg != self.global_size_A:
            sys = sc.sparse.eye(m=self.row_end - self.row_beg, n=self.global_size_A, k=self.row_beg) - self.dt * D[self.rank_subcol_alternating] * self.Apar
            h1_loc, it = self.linear_solver(sys, h_loc, x0, tol)
            # print(it, 'iterations on proc', self.rank)

        # case without spatial parallelization
        else:
            for i in range(self.Frac):
                sys = sc.sparse.eye(self.global_size_A) - self.dt * D[i + self.rank_col * self.Frac] * self.Apar
                if self.solver == 'custom':
                    h1_loc[i * self.global_size_A:(i + 1) * self.global_size_A], it = self.linear_solver(sys, h_loc[i * self.global_size_A:(i+1)*self.global_size_A], x0[i * self.global_size_A:(i + 1) * self.global_size_A], tol)
                    # print(it, 'iterations on proc', self.rank)
                else:
                    h1_loc[i * self.global_size_A:(i + 1) * self.global_size_A] = self.__linear_solver__(sys, h_loc[i * self.global_size_A:(i + 1) * self.global_size_A], x0[i * self.global_size_A:(i + 1) * self.global_size_A], tol)

        self.comm_col.Barrier()
        return h1_loc, it

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

    def __get_u_last__(self):

        # if self.time_intervals == 1:
        #     return np.inf

        err_max = 0

        # case with spatial parallelization, need reduction for maximal error
        if self.frac > 1:
            if self.size - self.size_subcol_seq <= self.rank:
                self.u_last_old_loc = self.u_last_loc.copy()
                self.u_last_loc = self.u_loc[:, -1]
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
                self.u_last_old_loc = self.u_last_loc.copy()
                self.u_last_loc = self.u_loc[-self.global_size_A:, -1]
                err_max = self.norm(self.u_last_old_loc - self.u_last_loc)

            # broadcast the error, a stopping criteria
            if self.size > 1:
                time_beg = MPI.Wtime()
                err_max = self.comm.bcast(err_max, root=self.size - 1)
                self.communication_time += MPI.Wtime() - time_beg

        return err_max

    def __bcast_u_last_loc__(self):

        if self.comm_last is not None and self.time_intervals > 1:# and self.size_col < self.size:
            time_beg = MPI.Wtime()
            self.u_last_loc = self.comm_last.bcast(self.u_last_loc, root=0)
            self.communication_time += MPI.Wtime() - time_beg

    def __write_time_in_txt__(self):
        if self.rank == 0:
            file = open(self.time_document, "w+")
            file.write(str(self.T_start) + '\n')
            for rolling_int in range(self.rolling):
                t_start = self.T_start + self.time_intervals * rolling_int * self.dt
                for k in range(self.time_intervals):
                    for i in range(self.time_points):
                        file.write(str(k * self.dt + self.t[i] + t_start) + '\n')
            file.close()

    def __write_u_in_txt__(self, rolling_interval):

        # with spatial parallelization
        if self.frac is not 0:
            if rolling_interval == 0:
                for proc in range(self.size_subcol_seq):
                    if self.rank == proc:
                        file = open(self.document, "a")
                        for element in self.u0_loc:
                            file.write(str(complex(element)) + ' ')
                        if (proc + 1) % self.frac is 0:
                            file.write('\n')
                        file.close()
                    self.comm.Barrier()

            for c in range(self.proc_row):
                for k in range(self.cols_loc):
                    for r in range(self.proc_col):
                        if self.rank_col is r and self.rank_row is c:
                            file = open(self.document, "a")
                            for element in self.u_loc[:, k]:
                                file.write(str(element) + ' ')
                            if (self.rank_col+1) % self.frac is 0:
                                file.write('\n')
                            file.close()
                        self.comm.Barrier()

        # without spatial parallelization
        else:
            if self.rank == 0:
                file = open(self.document, "a")
                for element in self.u0_loc:
                    file.write(str(complex(element)) + ' ')
                file.write('\n')
                file.close()
            self.comm.Barrier()

            for c in range(self.proc_row):
                for k in range(self.cols_loc):
                    for r in range(self.proc_col):
                        if self.rank_col is r and self.rank_row is c:
                            file = open(self.document, "a")
                            for i in range(self.Frac):
                                for element in self.u_loc[i*self.global_size_A:(i+1)*self.global_size_A, k]:
                                    file.write(str(element) + ' ')
                                file.write('\n')
                            file.close()
                        self.comm.Barrier()

    # solver (space parallelization not included yet)
    def __linear_solver__(self, M_loc, m_loc, m0, tol):

        # class gmres_counter(object):
        #     def __init__(self, disp=True):
        #         self._disp = disp
        #         self.niter = 0
        #
        #     def __call__(self, rk=None):
        #         self.niter += 1
        #         if self._disp:
        #             print('iter %3i\trk = %s' % (self.niter, str(rk)))
        # counter = gmres_counter()

        M = None
        m = None

        Solver = sc.sparse.linalg.spsolve
        if self.solver == 'gmres':
            Solver = sc.sparse.linalg.gmres

        if self.solver == 'gmres':
            x_loc, info = Solver(M_loc, m_loc, tol=tol, maxiter=self.smaxiter, x0=m0)
        else:
            x_loc = Solver(M_loc, m_loc)

        return x_loc

import numpy as np
from core_parallel.communicators import Communicators
from mpi4py import MPI
from scipy.sparse import linalg
import scipy as sc


class LinearHelpers(Communicators):

    def __init__(self):
        Communicators.__init__(self)

    def __next_alpha__(self, idx):
        if idx + 1 < len(self.alphas) and self.time_intervals > 1:
            idx += 1
        return idx

    def __get_v__(self, t_start):
        v = np.zeros(self.rows_loc, dtype=complex)
        shift = self.rank_row

        # case with spatial parallelization
        if self.frac > 1:
            for k in range(self.time_points):
                v += self.dt * self.Q[self.rank_subcol_alternating, k] * self.bpar(t_start + self.t[k] + shift * self.dt)

        # case without spatial parallelization
        else:
            for i in range(self.Frac):
                for k in range(self.time_points):
                    v[i * self.global_size_A:(i+1)*self.global_size_A] += self.dt * self.Q[i + self.Frac * self.rank_col, k] * self.bpar(t_start + self.t[k] + shift * self.dt)
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
    def __get_fft__(self, res_loc, a):

        if self.time_intervals == 1:
            return res_loc, '0'

        g_loc = a ** (self.rank_row / self.time_intervals) / self.time_intervals * res_loc    # scale
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

    def __get_residual__(self, v_loc):

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

        # assemble first part of res_loc
        if self.rank_row == 0:
            res_loc = v_loc.copy()
        else:
            # case with spatial parallelization
            if self.frac > 0:
                res_loc = v_loc + Hu_loc
            # case without spatial parallelization
            else:
                res_loc = v_loc + np.tile(Hu_loc, self.Frac)

        # computation of Au
        Au_loc = np.zeros_like(self.u_loc)
        # case with spatial parallelization
        if self.frac > 1:
            # TODO
            raise RuntimeError('Not implemented')

        # case without spatial parallelization
        else:
            for i in range(self.Frac):
                Au_loc[i * self.global_size_A: (i + 1) * self.global_size_A] = self.Apar @ self.u_loc[i * self.global_size_A: (i + 1) * self.global_size_A]

        # computation of Cu in Hu_loc
        Hu_loc = self.u_loc.copy(order='C')
        # case with spatial parallelization
        if self.frac > 1:
            # TODO
            raise RuntimeError('Not implemented')

        # case without spatial paralellization
        else:
            tmp = np.zeros_like(self.u_loc)

            # every processor reduces on p
            for p in range(self.size_col):
                tmp *= 0
                # compute local sums
                for i in range(self.Frac):
                    for j in range(self.Frac):
                        i_global = p * self.Frac + i
                        j_global = self.rank_col * self.Frac + j
                        tmp[i * self.global_size_A: (i + 1) * self.global_size_A] -= self.dt * self.Q[i_global, j_global] * Au_loc[j * self.global_size_A: (j + 1) * self.global_size_A]

                # reduce to p
                tmp2 = None
                if self.size_col > 1:
                    time_beg = MPI.Wtime()
                    tmp2 = self.comm_col.reduce(tmp, op=MPI.SUM, root=p)
                    self.communication_time += MPI.Wtime() - time_beg

                if self.rank_col == p:
                    Hu_loc += tmp
                elif tmp2 is not None:
                    Hu_loc += tmp2

        res_loc -= Hu_loc

        # add u0 where needed
        if self.rank_row == 0:
            # with spatial parallelization
            if self.frac > 1:
                res_loc += self.u0_loc

            # without spatial parallelization
            else:
                res_loc += np.tile(self.u0_loc, self.Frac)

        return res_loc

    def __step1__(self, Zinv, g_loc):

        h_loc = np.zeros_like(g_loc, dtype=complex)

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

        it = 0
        h1_loc = np.zeros_like(h_loc, dtype=complex, order='C')
        # case with spatial parallelization
        if self.row_end - self.row_beg != self.global_size_A:
            sys = sc.sparse.eye(m=self.row_end - self.row_beg, n=self.global_size_A, k=self.row_beg) - self.dt * D[self.rank_subcol_alternating] * self.Apar
            h1_loc, it = self.linear_solver(sys, h_loc, x0, tol)
            #print(it, 'iterations on proc', self.rank)

        # case without spatial parallelization
        else:
            for i in range(self.Frac):
                sys = sc.sparse.eye(self.global_size_A) - self.dt * D[i + self.rank_col * self.Frac] * self.Apar
                print(self.rank, linalg.sparse.expm_cond(sys))
                if self.solver == 'custom':
                    h1_loc[i * self.global_size_A:(i + 1) * self.global_size_A], it = self.linear_solver(sys, h_loc[i * self.global_size_A:(i + 1) * self.global_size_A], x0[i * self.global_size_A:(i + 1) * self.global_size_A], tol)
                    #print(it, 'iterations on proc', self.rank)
                else:
                    h1_loc[i * self.global_size_A:(i + 1) * self.global_size_A] = self.__linear_solver__(sys, h_loc[i * self.global_size_A:(i + 1) * self.global_size_A], x0[i * self.global_size_A:(i + 1) * self.global_size_A], tol)

        self.comm_col.Barrier()
        return h1_loc, it

    # ifft
    def __get_ifft__(self, h1_loc, a):

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
            req = self.comm_row.isend(h1_loc, dest=comm_with, tag=k)
            hr = self.comm_row.recv(source=comm_with, tag=k)
            req.Wait()
            self.communication_time += MPI.Wtime() - time_beg

            # glue the info
            h1_loc = hr + factor * h1_loc

            # scale the output
            if R[k] == '1' and scalar != 1:
                h1_loc *= scalar

        h1_loc *= a**(-self.rank_row / self.time_intervals)

        return h1_loc

    def __get_max_norm__(self, c):

        err_loc = self.norm(c)
        time_beg = MPI.Wtime()
        err_max = self.comm.allreduce(err_loc, op=MPI.MAX)
        self.communication_time += MPI.Wtime() - time_beg

        return err_max

    def __bcast_u_last_loc__(self):

        if self.comm_last != None and self.time_intervals > 1:# and self.size_col < self.size:
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
        if self.frac != 0:
            if rolling_interval == 0:
                for proc in range(self.size_subcol_seq):
                    if self.rank == proc:
                        file = open(self.document, "a")
                        for element in self.u0_loc:
                            file.write(str(complex(element)) + ' ')
                        if (proc + 1) % self.frac == 0:
                            file.write('\n')
                        file.close()
                    self.comm.Barrier()

            for c in range(self.proc_row):
                for r in range(self.proc_col):
                    if self.rank_col == r and self.rank_row == c:
                        file = open(self.document, "a")
                        for element in self.u_loc:
                            file.write(str(element) + ' ')
                        if (self.rank_col+1) % self.frac == 0:
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
                for r in range(self.proc_col):
                    if self.rank_col == r and self.rank_row == c:
                        file = open(self.document, "a")
                        for i in range(self.Frac):
                            for element in self.u_loc[i*self.global_size_A:(i+1)*self.global_size_A]:
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

        Solver = linalg.spsolve
        if self.solver == 'gmres':
            Solver = linalg.gmres

        if self.solver == 'gmres':
            x_loc, info = Solver(M_loc, m_loc, tol=tol, atol=0, maxiter=self.smaxiter, x0=m0)
        else:
            x_loc = Solver(M_loc, m_loc)

        return x_loc

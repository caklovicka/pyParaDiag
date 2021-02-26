import numpy as np
from scipy import sparse
from core_parallel.linear_paralpha import LinearParalpha
import scipy as sc
from petsc4py import PETSc

"""
schrodinger in 2d, 2nd order in space
u_t = c ( u_xx + u_yy )
"""


class Schrodinger(LinearParalpha):

    # user defined, just for this class
    c = 1j
    sigma = 1
    p = [1, 1]
    X_left = -10
    X_right = 10
    Y_left = -10
    Y_right = 10
    xx = None
    yy = None

    # PROCESSORS NEEDED = proc_row * proc_col

    def __init__(self):

        LinearParalpha.__init__(self)

    def setup(self):

        # ---- PRESETUP ----

        self.xx = np.linspace(self.X_left, self.X_right, self.spatial_points[0])
        self.yy = np.linspace(self.Y_left, self.Y_right, self.spatial_points[1])

        self.dx = []
        self.dx.append((self.X_right - self.X_left) / (self.spatial_points[0] - 1))
        self.dx.append((self.Y_right - self.Y_left) / (self.spatial_points[1] - 1))

        # x and size_global_A have to be filled before super().setup()

        self.x = np.meshgrid(self.xx, self.yy)
        del self.xx, self.yy

        self.global_size_A = 1
        for n in self.spatial_points:
            self.global_size_A *= n

        # ---- PRESETUP <end> ----

        # do not delete this, this builds communicators needed for your matrix Apar
        super().setup()

        # ---- POSTSETUP ----

        # user defined, parallel matrix, rows are divided between processors in group 'comm_matrix'
        # rows_beg and rows_end define the chunk of matrix Apar in this communicator

        # user defined matrix Apar, sparse form recommended
        row = list()
        col = list()
        data = list()
        cx = self.c / self.dx[0]**2
        cy = self.c / self.dx[1]**2

        # 15/4	−77/6	107/6	−13	   61/12	 −5/6
        for i in range(self.row_beg, self.row_end, 1):
            # x part
            row.append(i)
            col.append(i)
            data.append(15/4 * cx + 15/4 * cy)

            if i % self.spatial_points[0] + 1 < self.spatial_points[0]:
                row.append(i)
                col.append(i + 1)
                data.append(-77/6 * cx)

            if i % self.spatial_points[0] + 2 < self.spatial_points[0]:
                row.append(i)
                col.append(i + 2)
                data.append(107/6 * cx)

            if i % self.spatial_points[0] + 3 < self.spatial_points[0]:
                row.append(i)
                col.append(i + 3)
                data.append(-13 * cx)

            if i % self.spatial_points[0] + 4 < self.spatial_points[0]:
                row.append(i)
                col.append(i + 4)
                data.append(61/12 * cx)

            if i % self.spatial_points[0] + 5 < self.spatial_points[0]:
                row.append(i)
                col.append(i + 5)
                data.append(-5/6 * cx)

            # y part
            if i + self.spatial_points[0] < self.global_size_A:
                row.append(i)
                col.append(i + self.spatial_points[0])
                data.append(-77/6 * cy)

            if i + 2 * self.spatial_points[0] < self.global_size_A:
                row.append(i)
                col.append(i + 2 * self.spatial_points[0])
                data.append(107/6 * cy)

            if i + 3 * self.spatial_points[0] < self.global_size_A:
                row.append(i)
                col.append(i + 3 * self.spatial_points[0])
                data.append(-13 * cy)

            if i + 4 * self.spatial_points[0] < self.global_size_A:
                row.append(i)
                col.append(i + 4 * self.spatial_points[0])
                data.append(61/12 * cy)

            if i + 5 * self.spatial_points[0] < self.global_size_A:
                row.append(i)
                col.append(i + 5 * self.spatial_points[0])
                data.append(-5/6 * cy)

        data = np.array(data)
        row = np.array(row) - self.row_beg
        col = np.array(col)
        self.Apar = sparse.csr_matrix((data, (row, col)), shape=(self.row_end - self.row_beg, self.global_size_A))

        del data, row, col

        # ---- POSTSETUP <end> ----

    # user defined
    def bpar(self, t):
        return self.rhs(t, self.x).flatten()[self.row_beg:self.row_end]

    # user defined
    def u_exact(self, t, z):
        a = self.sigma + 2j * t
        const = self.sigma / a * np.exp(self.sigma/2 * (self.p[0]**2 + self.p[1]**2) * (self.sigma/a - 1))
        exp1 = np.exp(1j * self.sigma / a * (self.p[0] * z[0] + self.p[1] * z[1]))
        exp2 = np.exp(-1/(2 * a) * (z[0]**2 + z[1]**2))
        return const * exp1 * exp2

    # DELETE
    def u_t(self, t, z):
        a = self.sigma + 2j * t
        exp_const1 = self.sigma/2 * (self.p[0]**2 + self.p[1]**2) * (self.sigma/a - 1)
        exp_const2 = 1j * self.sigma / a * (self.p[0] * z[0] + self.p[1] * z[1])
        exp_const3 = -1/(2 * a) * (z[0]**2 + z[1]**2)
        exp = np.exp(exp_const1 + exp_const2 + exp_const3)

        sum1 = -2j * self.sigma / a**2 * exp
        sum2 = self.sigma/a * exp * (-self.sigma**2 * 2j / a**2 + (z[0]**2 + z[1]**2) * 1j / a**2 - 1j * self.sigma * 2j/a**2)

        return sum1 + sum2

    # user defined
    def u_initial(self, z):
        return self.u_exact(self.T_start, z)

    # user defined
    def rhs(self, t, z):
        return 0 * z[0] * z[1]

    @staticmethod
    def norm(x):
        return np.linalg.norm(x, np.inf)

    # petsc solver on comm_matrix
    def linear_solver(self, M_loc, m_loc, m0_loc, tol):
        m = PETSc.Vec()
        m.createWithArray(array=m_loc, comm=self.comm_matrix)
        m0 = PETSc.Vec()
        m0.createWithArray(array=m0_loc, comm=self.comm_matrix)
        M = PETSc.Mat()
        csr = (M_loc.indptr, M_loc.indices, M_loc.data)
        M.createAIJ(size=(self.global_size_A, self.global_size_A), csr=csr, comm=self.comm_matrix)

        ksp = PETSc.KSP()
        ksp.create()
        ksp.setType('gmres')
        ksp.setFromOptions()
        ksp.setTolerances(rtol=tol, max_it=self.smaxiter)
        pc = ksp.getPC()
        pc.setType('none')
        ksp.setOperators(M)
        ksp.setInitialGuessNonzero(True)
        ksp.solve(m, m0)
        sol = m0.getArray()
        it = ksp.getIterationNumber()

        m.destroy()
        m0.destroy()
        ksp.destroy()
        M.destroy()

        return sol, it

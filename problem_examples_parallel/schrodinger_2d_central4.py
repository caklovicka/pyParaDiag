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

        self.xx = np.linspace(self.X_left, self.X_right, self.spatial_points[0] + 2)[1:-1]
        self.yy = np.linspace(self.Y_left, self.Y_right, self.spatial_points[1] + 2)[1:-1]

        self.dx = []
        self.dx.append((self.X_right - self.X_left) / (self.spatial_points[0] + 1))
        self.dx.append((self.Y_right - self.Y_left) / (self.spatial_points[1] + 1))

        # x and size_global_A have to be filled before super().setup()

        self.x = np.meshgrid(self.xx, self.yy)

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
        cx = self.c / (12 * self.dx[0]**2)
        cy = self.c / (12 * self.dx[1]**2)
        for i in range(self.row_beg, self.row_end, 1):

            # for x with y on diagonal
            if i % self.spatial_points[0] == 0:
                row.append(i)
                col.append(i)
                data.append(-15 * cx)
                if i < self.spatial_points[0] or i >= self.global_size_A - self.spatial_points[0]:
                    data[-1] -= 15 * cy
                else:
                    data[-1] -= 30 * cy

                row.append(i)
                col.append(i + 1)
                data.append(-4 * cx)
                row.append(i)
                col.append(i + 2)
                data.append(14 * cx)
                row.append(i)
                col.append(i + 3)
                data.append(-6 * cx)
                row.append(i)
                col.append(i + 4)
                data.append(cx)

            elif i % self.spatial_points[0] == 1:
                row.append(i)
                col.append(i - 1)
                data.append(16 * cx)
                row.append(i)
                col.append(i)
                data.append(-30 * cx)
                if i < self.spatial_points[0] or i >= self.global_size_A - self.spatial_points[0]:
                    data[-1] -= 15 * cy
                else:
                    data[-1] -= 30 * cy
                row.append(i)
                col.append(i + 1)
                data.append(16 * cx)
                row.append(i)
                col.append(i + 2)
                data.append(-cx)

            elif i % self.spatial_points[0] == self.spatial_points[0] - 2:
                row.append(i)
                col.append(i - 2)
                data.append(-cx)
                row.append(i)
                col.append(i - 1)
                data.append(16 * cx)
                row.append(i)
                col.append(i)
                data.append(-30 * cx)
                if i < self.spatial_points[0] or i >= self.global_size_A - self.spatial_points[0]:
                    data[-1] -= 15 * cy
                else:
                    data[-1] -= 30 * cy
                row.append(i)
                col.append(i + 1)
                data.append(16 * cx)

            elif i % self.spatial_points[0] == self.spatial_points[0] - 1:
                row.append(i)
                col.append(i - 4)
                data.append(cx)
                row.append(i)
                col.append(i - 3)
                data.append(-6 * cx)
                row.append(i)
                col.append(i - 2)
                data.append(14 * cx)
                row.append(i)
                col.append(i - 1)
                data.append(-4 * cx)
                row.append(i)
                col.append(i)
                data.append(-15 * cx)
                if i < self.spatial_points[0] or i >= self.global_size_A - self.spatial_points[0]:
                    data[-1] -= 15 * cy
                else:
                    data[-1] -= 30 * cy

            else:
                row.append(i)
                col.append(i - 2)
                data.append(-cx)
                row.append(i)
                col.append(i - 1)
                data.append(16 * cx)
                row.append(i)
                col.append(i)
                data.append(-30 * cx)
                if i < self.spatial_points[0] or i >= self.global_size_A - self.spatial_points[0]:
                    data[-1] -= 15 * cy
                else:
                    data[-1] -= 30 * cy
                row.append(i)
                col.append(i + 1)
                data.append(16 * cx)
                row.append(i)
                col.append(i + 2)
                data.append(-cx)

            # for y outside diagonal
            if i < self.spatial_points[0]:
                row.append(i)
                col.append(i + self.spatial_points[0])
                data.append(-4 * cy)
                row.append(i)
                col.append(i + 2 * self.spatial_points[0])
                data.append(14 * cy)
                row.append(i)
                col.append(i + 3 * self.spatial_points[0])
                data.append(-6 * cy)
                row.append(i)
                col.append(i + 4 * self.spatial_points[0])
                data.append(cy)

            elif self.spatial_points[0] <= i < 2 * self.spatial_points[0]:
                row.append(i)
                col.append(i - self.spatial_points[0])
                data.append(16 * cy)
                row.append(i)
                col.append(i + self.spatial_points[0])
                data.append(16 * cy)
                row.append(i)
                col.append(i + 2 * self.spatial_points[0])
                data.append(-cy)

            elif self.global_size_A - 2 * self.spatial_points[0] <= i < self.global_size_A - self.spatial_points[0]:
                row.append(i)
                col.append(i - 2 * self.spatial_points[0])
                data.append(-cy)
                row.append(i)
                col.append(i - self.spatial_points[0])
                data.append(16 * cy)
                row.append(i)
                col.append(i + self.spatial_points[0])
                data.append(16 * cy)

            elif self.global_size_A - self.spatial_points[0] <= i:
                row.append(i)
                col.append(i - 4 * self.spatial_points[0])
                data.append(cy)
                row.append(i)
                col.append(i - 3 * self.spatial_points[0])
                data.append(-6 * cy)
                row.append(i)
                col.append(i - 2 * self.spatial_points[0])
                data.append(14 * cy)
                row.append(i)
                col.append(i - self.spatial_points[0])
                data.append(-4 * cy)

            else:
                row.append(i)
                col.append(i - 2 * self.spatial_points[0])
                data.append(-cy)
                row.append(i)
                col.append(i - self.spatial_points[0])
                data.append(16 * cy)
                row.append(i)
                col.append(i + self.spatial_points[0])
                data.append(16 * cy)
                row.append(i)
                col.append(i + 2 * self.spatial_points[0])
                data.append(-cy)

        data = np.array(data)
        row = np.array(row) - self.row_beg
        col = np.array(col)
        self.Apar = sparse.csr_matrix((data, (row, col)), shape=(self.row_end - self.row_beg, self.global_size_A))

        del data, row, col

        # ---- POSTSETUP <end> ----

    # user defined
    def bpar(self, t):
        cx = self.c / (12 * self.dx[0]**2)
        cy = self.c / (12 * self.dx[1]**2)
        bb = np.zeros(self.global_size_A, dtype=complex)
        bb += self.rhs(t, self.x).flatten()

        bb[::self.spatial_points[0]] += 10 * cx * self.a(t, self.xx)
        bb[1::self.spatial_points[0]] += -cx * self.a(t, self.xx)

        bb[self.spatial_points[0] - 2::self.spatial_points[0]] += -cx * self.b(t, self.xx)
        bb[self.spatial_points[0] - 1::self.spatial_points[0]] += 10 * cx * self.b(t, self.xx)

        bb[:self.spatial_points[0]] += 10 * cy * self.e(t, self.yy)
        bb[self.spatial_points[0]:2 * self.spatial_points[0]] += -cy * self.e(t, self.yy)

        bb[-2 * self.spatial_points[0]:-self.spatial_points[0]] += -cy * self.d(t, self.yy)
        bb[-self.spatial_points[0]:] += 10 * cy * self.d(t, self.yy)
        return bb[self.row_beg:self.row_end]

    # user defined
    def u_exact(self, t, z):
        a = self.sigma + 2j * t
        const = self.sigma / a * np.exp(self.sigma/2 * (self.p[0]**2 + self.p[1]**2) * (self.sigma/a - 1))
        exp1 = np.exp(1j * self.sigma / a * (self.p[0] * z[0] + self.p[1] * z[1]))
        exp2 = np.exp(-1/(2 * a) * (z[0]**2 + z[1]**2))
        return const * exp1 * exp2

    # user defined
    def a(self, t, y):
        return self.u_exact(t, (self.X_left, y))

    # user defined
    def b(self, t, y):
        return self.u_exact(t, (self.X_right, y))

    # user defined
    def e(self, t, x):
        return self.u_exact(t, (x, self.Y_left))

    # user defined
    def d(self, t, x):
        return self.u_exact(t, (x, self.Y_right))

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
        ksp.setTolerances(rtol=tol, max_it=self.global_size_A)
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

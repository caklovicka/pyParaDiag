import numpy as np
from scipy import sparse
from petsc4py import PETSc
from core_parallel.linear_paralpha import LinearParalpha

"""
advection eq. in 2d
u_t + c_x * u_x + c_y * u_y = f
"""


class Advection(LinearParalpha):

    # user defined, just for this class
    c = [1, 1]
    X_left = 0
    X_right = 1
    Y_left = 0
    Y_right = 1
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
        for i in range(self.row_beg, self.row_end, 1):
            # right x diagonal
            if (i+1) % self.spatial_points[0] != 0:
                row.append(i)
                col.append(i+1)
                data.append(-self.c[0] / (2 * self.dx[0]))
            # left x diagonal
            if i % self.spatial_points[0] != 0:
                row.append(i)
                col.append(i - 1)
                data.append(self.c[0] / (2 * self.dx[0]))

            # left y diagonal
            if i >= self.spatial_points[0]:
                row.append(i)
                col.append(i-self.spatial_points[0])
                data.append(self.c[1] / (2 * self.dx[1]))
            # right y diagonal
            if i + self.spatial_points[0] < self.global_size_A:
                row.append(i)
                col.append(i+self.spatial_points[0])
                data.append(-self.c[1] / (2 * self.dx[1]))

        data = np.array(data)
        row = np.array(row) - self.row_beg
        col = np.array(col)
        self.Apar = sparse.csr_matrix((data, (row, col)), shape=(self.row_end - self.row_beg, self.global_size_A))

        # ---- POSTSETUP <end> ----

    # user defined
    def bpar(self, t):

        bb = self.rhs(t, self.x).flatten()
        bb[:self.spatial_points[0]] += self.c[1] / (2 * self.dx[1]) * self.e(t, self.xx)
        bb[-self.spatial_points[0]:] -= self.c[1] / (2 * self.dx[1]) * self.d(t, self.xx)
        bb[::self.spatial_points[0]] += self.c[0] / (2 * self.dx[0]) * self.a(t, self.yy)
        bb[self.spatial_points[0]-1::self.spatial_points[0]] -= self.c[0] / (2 * self.dx[0]) * self.b(t, self.yy)

        return bb[self.row_beg:self.row_end]

    # user defined
    def u_exact(self, t, z):
        return np.sin(2 * np.pi * (z[0] - self.c[0] * t)) * np.sin(2 * np.pi * (z[1] - self.c[1] * t))

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
    @staticmethod
    def rhs(t, z):
        return 0 * z[0] * z[1]

    @staticmethod
    def norm(x):
        return np.linalg.norm(x.real, np.inf)

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
        ksp.setTolerances(rtol=tol, atol=tol, max_it=self.global_size_A)
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








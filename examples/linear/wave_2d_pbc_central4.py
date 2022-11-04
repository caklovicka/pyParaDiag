import numpy as np
from scipy import sparse
from core.linear_paralpha import LinearParalpha
from petsc4py import PETSc

"""
wave eq. in 2d, 4th order discretization in space
u_tt = c ( u_xx + u_yy ) + f
"""


class Wave(LinearParalpha):

    # user defined, just for this class
    c = 1
    # take care of periodic boundary conditions
    X_left = 0
    X_right = 2
    Y_left = 0
    Y_right = 2
    xx = None
    yy = None

    # PROCESSORS NEEDED = proc_row * proc_col

    def __init__(self):

        super().__init__()

    def setup(self):

        # ---- PRESETUP ----

        self.xx = np.linspace(self.X_left, self.X_right, self.spatial_points[0] + 1)[:-1]
        self.yy = np.linspace(self.Y_left, self.Y_right, self.spatial_points[1] + 1)[:-1]

        self.dx = []
        self.dx.append((self.X_right - self.X_left) / (self.spatial_points[0]))
        self.dx.append((self.Y_right - self.Y_left) / (self.spatial_points[1]))

        # x and size_global_A have to be filled before super().setup()

        self.x = np.meshgrid(self.xx, self.yy)

        self.global_size_A = 1
        for n in self.spatial_points:
            self.global_size_A *= n
        self.global_size_A *= 2

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
        cx = self.c**2 / (12 * self.dx[0] ** 2)
        cy = self.c**2 / (12 * self.dx[1] ** 2)
        for i in range(self.row_beg, self.row_end, 1):

            # this is sparse eye
            if i < self.global_size_A // 2:
                row.append(i)
                col.append(i + self.global_size_A // 2)
                data.append(1)

            else:
                # x part
                row.append(i)
                col.append(i - self.global_size_A // 2)
                data.append(-30 * cx - 30 * cy)

                row.append(i)
                col.append((i + 1) % self.spatial_points[0] + (i // self.spatial_points[0]) * self.spatial_points[0] - self.global_size_A // 2)
                data.append(16 * cx)

                row.append(i)
                col.append((i + 2) % self.spatial_points[0] + (i // self.spatial_points[0]) * self.spatial_points[0] - self.global_size_A // 2)
                data.append(-cx)

                row.append(i)
                col.append((i - 1) % self.spatial_points[0] + (i // self.spatial_points[0]) * self.spatial_points[0] - self.global_size_A // 2)
                data.append(16 * cx)

                row.append(i)
                col.append((i - 2) % self.spatial_points[0] + (i // self.spatial_points[0]) * self.spatial_points[0] - self.global_size_A // 2)
                data.append(-cx)

                # y part
                row.append(i)
                col.append((i + self.spatial_points[0]) % (self.global_size_A // 2))
                data.append(16 * cy)

                row.append(i)
                col.append((i + 2 * self.spatial_points[0]) % (self.global_size_A // 2))
                data.append(-cy)

                row.append(i)
                col.append((i - self.spatial_points[0]) % (self.global_size_A // 2))
                data.append(16 * cy)

                row.append(i)
                col.append((i - 2 * self.spatial_points[0]) % (self.global_size_A // 2))
                data.append(-cy)

        data = np.array(data)
        row = np.array(row) - self.row_beg
        col = np.array(col)
        self.Apar = sparse.csr_matrix((data, (row, col)), shape=(self.row_end - self.row_beg, self.global_size_A))

        # ---- POSTSETUP <end> ----

    # user defined
    def bpar(self, t):

        bb = np.zeros(self.global_size_A)
        bb[self.global_size_A // 2:] = self.rhs(t, self.x).flatten()
        return bb[self.row_beg:self.row_end]

    # user defined
    def u_exact(self, t, z):
        return np.sin(np.pi * t) * np.sin(self.c * np.pi * z[0]) * np.sin(self.c * np.pi * z[1])

    # user defined
    def u_initial_0(self, z):
        return self.u_exact(self.T_start, z)

    # user defined
    def u_initial_1(self, z):
        return np.pi * np.cos(np.pi * self.T_start) * np.sin(self.c * np.pi * z[0]) * np.sin(self.c * np.pi * z[1])

    # user defined
    def u_initial(self, z):
        return np.array([self.u_initial_0(z), self.u_initial_1(z)])

    # user defined
    def rhs(self, t, z):
        return 0 * z[1] * z[0]

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
        M.createAIJWithArrays(size=(self.global_size_A, self.global_size_A), csr=csr, comm=self.comm_matrix)

        ksp = PETSc.KSP()
        ksp.create(comm=self.comm_matrix)
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
        pc.destroy()

        return sol, it

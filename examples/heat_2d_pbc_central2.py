import numpy as np
from scipy import sparse
from core.imex_newton_refinement import PartiallyCoupled
from petsc4py import PETSc

"""
heat1 eq. in 2d, 2nd order central differences
y_t = c ( y_xx + y_yy ) + f
"""


class Heat(PartiallyCoupled):

    # user defined, just for this class
    c = 1
    gamma = 0.05

    T_start = 0
    T_end = 0.1

    X_left = 0
    X_right = 1
    Y_left = 0
    Y_right = 1
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

        # ---- PRESETUP <end> ----

        # do not delete this, this builds communicators needed for your matrix Apar
        super().setup()

        # ---- POSTSETUP ----

        # # user defined, parallel matrix, rows are divided between processors in group 'comm_matrix'
        # rows_beg and rows_end define the chunk of matrix Apar in this communicator

        # user defined matrix Apar, sparse form
        # it is a same matrix for state and adjoint
        # in case of a different ones, use variables self.state and self.adjoint to determine which matrix to define
        row = list()
        col = list()
        data = list()
        for i in range(self.row_beg, self.row_end, 1):

            row.append(i)
            col.append(i)
            data.append(-2 * self.c * (1/self.dx[0]**2 + 1/self.dx[1]**2))

            row.append(i)
            col.append((i + 1) % self.spatial_points[0] + (i // self.spatial_points[0]) * self.spatial_points[0])
            data.append(self.c / self.dx[0] ** 2)

            row.append(i)
            col.append((i - 1) % self.spatial_points[0] + (i // self.spatial_points[0]) * self.spatial_points[0])
            data.append(self.c / self.dx[0] ** 2)

            row.append(i)
            col.append((i + self.spatial_points[0]) % self.global_size_A)
            data.append(self.c / self.dx[1] ** 2)

            row.append(i)
            col.append((i - self.spatial_points[0]) % self.global_size_A)
            data.append(self.c / self.dx[1] ** 2)

        data = np.array(data)
        row = np.array(row) - self.row_beg
        col = np.array(col)
        self.Apar = sparse.csr_matrix((data, (row, col)), shape=(self.row_end - self.row_beg, self.global_size_A))

        # ---- POSTSETUP <end> ----

    def bpar_y(self, t):
        return np.zeros(self.row_end - self.row_end)

    # user defined
    def bpar_p(self, t):
        return np.zeros(self.row_end - self.row_end)

    def y_initial(self, x):
        return self.y(self.T_start, x)

    def p_end(self, x):
        return self.p(self.T_end, x)

    def yd(self, t, x):
        return ((2 * np.pi ** 2 / 4 + 2 / np.pi ** 2 / self.gamma) * np.exp(self.T_end) +
                (1 - np.pi ** 2 / 2 - 4 / (4 + 2 * np.pi ** 2) / self.gamma) * np.exp(t)) * \
                np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)

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

    # exact solutions, not necessary for this class
    def y(self, t, x):
        return (2 / (np.pi ** 2 * self.gamma) * np.exp(self.T_end) - 4 / (4 + 2 * np.pi ** 2) / self.gamma *
                np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)

    def p(self, t, x):
        return (np.exp(self.T_end) - np.exp(t)) * np.cos(np.pi * x[0] / 2) * np.cos(np.pi * x[1] / 2)

    @staticmethod
    def u_exact(t, x):
        return 1 / gamma * p(t, x)

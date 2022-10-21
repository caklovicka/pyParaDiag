import numpy as np
from scipy import sparse
from core_parallel.nonlinear_increment_paralpha import NonlinearParalpha
from petsc4py import PETSc

"""
Allen Cahn eq. in 2d, 2nd order central differences and pbc
u_t = c ( u_xx + u_yy ) + f
"""


class AllenCahn(NonlinearParalpha):

    # user defined, just for this class
    c = 1
    R = 0.25
    eps = 0.04
    X_left = -0.5
    X_right = 0.5
    Y_left = -0.5
    Y_right = 0.5
    xx = None
    yy = None

    # PROCESSORS NEEDED = proc_row * proc_col

    def __init__(self):

        NonlinearParalpha.__init__(self)

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

        # user defined matrix Apar, sparse form recommended
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

    # user defined - returns local chunk, depends on u
    def F(self, u):
        return 1 / self.eps ** 2 * u * (1 - u ** 2)

    # user defined - returns local chunk, depends on u
    # for now, depends just on u and assumes the Jacobian is a diagonal matrix
    def dF(self, u):
        return 1 / self.eps ** 2 * (1 - 3 * u ** 2)

    # user defined
    def u_initial(self, z):
        return np.tanh((self.R - np.sqrt(z[0] ** 2 + z[1] ** 2)) / (np.sqrt(2) * self.eps))

    # user defined, depends on (t, x)
    @staticmethod
    def rhs(t, z):
        return 0 * z[0] * z[1]

    def bpar(self, t):
        return self.rhs(t, self.x).flatten()[self.row_beg:self.row_end]

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
        ksp.create(comm=self.comm_matrix)
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

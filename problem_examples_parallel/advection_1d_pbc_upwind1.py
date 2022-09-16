import numpy as np
from scipy import sparse
from petsc4py import PETSc
from core_parallel.linear_paralpha import LinearParalpha

"""
advection1 eq. in 2d
u_t + c_x * u_x + c_y * u_y = f
"""


class Advection(LinearParalpha):

    # user defined, just for this class
    c = [0.1]
    X_left = 0
    X_right = 1
    xx = None
    yy = None

    # PROCESSORS NEEDED = proc_row * proc_col

    def __init__(self):

        LinearParalpha.__init__(self)

    def setup(self):

        # ---- PRESETUP ----

        self.x = np.linspace(self.X_left, self.X_right, self.spatial_points[0] + 1)[:-1]

        self.dx = []
        self.dx.append((self.X_right - self.X_left) / self.spatial_points[0])

        # x and size_global_A have to be filled before super().setup()
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
        cx = -self.c[0] / self.dx[0]
        for i in range(self.row_beg, self.row_end, 1):

            row.append(i)
            col.append((i - 1) % self.spatial_points[0])
            data.append(-cx)

            row.append(i)
            col.append(i % self.spatial_points[0])
            data.append(cx)

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
        return np.sin(2 * np.pi * (z - self.c[0] * t))

    # user defined
    def u_initial(self, z):
        return self.u_exact(self.T_start, z)

    # user defined
    @staticmethod
    def rhs(t, z):
        return 0 * z[0]

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








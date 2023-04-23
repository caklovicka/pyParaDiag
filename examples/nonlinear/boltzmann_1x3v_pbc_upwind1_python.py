import numpy as np
from scipy import sparse
from petsc4py import PETSc
from core.imex_newton_refinement import IMEXNewtonIncrementParalpha
from mpi4py import MPI
import ast

"""
boltzman equation
u_t + v1 * u_x = 1 / knudsen * Q(u)
"""


class Boltzmann(IMEXNewtonIncrementParalpha):

    # user defined, just for this class
    X_left = 0
    X_right = 1

    L = 8
    U_left = -L
    U_right = L
    V_left = -L
    V_right = L
    W_left = -L
    W_right = L

    knudsen = 1e-2
    xx = None
    uu = None
    vv = None
    ww = None

    # these HAVE to be python objects
    gas_fsm_kn = None       # = gas.fsm.Kn
    gas_fsm_nm = None       # = gas.fsm.nm
    gas_gamma = None        # = gas.γ
    phi = None
    psi = None
    chi = None

    # PROCESSORS NEEDED = proc_row * proc_col

    def __init__(self):

        super().__init__()

    def setup(self):

        # ---- PRESETUP ----

        # red gas_fsm_kn
        file = open('../../../../../../../examples/nonlinear/julia_setups/gas_fsm_kn.txt', 'r')
        #file = open('../../../examples/nonlinear/julia_setups/gas_fsm_kn.txt', 'r')
        self.gas_fsm_kn = float(file.read())
        file.close()

        # read gas_fsm_nm
        file = open('../../../../../../../examples/nonlinear/julia_setups/gas_fsm_nm.txt', 'r')
        #file = open('../../../examples/nonlinear/julia_setups/gas_fsm_nm.txt', 'r')
        self.gas_fsm_nm = int(file.read())
        file.close()

        # read gas_gamma
        file = open('../../../../../../../examples/nonlinear/julia_setups/gas_gamma.txt', 'r')
        #file = open('../../../examples/nonlinear/julia_setups/gas_gamma.txt', 'r')
        self.gas_gamma = float(file.read())
        file.close()

        # read phi
        file = open('../../../../../../../examples/nonlinear/julia_setups/phi.txt', 'r')
        #file = open('../../../examples/nonlinear/julia_setups/phi.txt', 'r')
        shape = list(map(int, file.readline().split()))
        self.phi = np.transpose(np.array(ast.literal_eval(file.read())).reshape(shape[::-1]), axes=(3, 2, 1, 0))
        file.close()

        # read psi
        file = open('../../../../../../../examples/nonlinear/julia_setups/psi.txt', 'r')
        #file = open('../../../examples/nonlinear/julia_setups/psi.txt', 'r')
        shape = list(map(int, file.readline().split()))
        self.psi = np.transpose(np.array(ast.literal_eval(file.read())).reshape(shape[::-1]), axes=(3, 2, 1, 0))
        file.close()

        # read chi
        file = open('../../../../../../../examples/nonlinear/julia_setups/chi.txt', 'r')
        #file = open('../../../examples/nonlinear/julia_setups/chi.txt', 'r')
        shape = list(map(int, file.readline().split()))
        self.chi = np.transpose(np.array(ast.literal_eval(file.read())).reshape(shape[::-1]), axes=(2, 1, 0))
        file.close()

        # step size
        self.dx = []
        self.dx.append((self.X_right - self.X_left) / (self.spatial_points[0] - 1))
        self.dx.append((self.U_right - self.U_left) / self.spatial_points[1])
        self.dx.append((self.V_right - self.V_left) / self.spatial_points[2])
        self.dx.append((self.W_right - self.W_left) / self.spatial_points[3])

        # mesh
        self.xx = np.arange(self.X_left - self.dx[0] / 2, self.X_right + self.dx[0], self.dx[0])[1:]
        self.uu = np.arange(self.U_left + self.dx[1] / 2, self.U_right + self.dx[1] / 2, self.dx[1])
        self.vv = np.arange(self.V_left + self.dx[2] / 2, self.V_right + self.dx[2] / 2, self.dx[2])
        self.ww = np.arange(self.W_left + self.dx[3] / 2, self.W_right + self.dx[3] / 2, self.dx[3])

        # x and size_global_A have to be filled before super().setup()
        self.x = np.meshgrid(self.xx, self.uu, self.vv, self.ww)
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
        Nuvw = self.spatial_points[1] * self.spatial_points[2] * self.spatial_points[3]
        Nvw = self.spatial_points[2] * self.spatial_points[3]

        for i in range(self.row_beg, self.row_end, 1):

            iu = (i % Nuvw) // Nvw

            if self.uu[iu] < 0:
                row.append(i)
                col.append(i)
                data.append(self.uu[iu] / self.dx[0])

                row.append(i)
                col.append((i + Nuvw) % self.global_size_A)
                data.append(-self.uu[iu] / self.dx[0])

            else:
                row.append(i)
                col.append(i)
                data.append(-self.uu[iu] / self.dx[0])

                row.append(i)
                col.append((i - Nuvw) % self.global_size_A)
                data.append(self.uu[iu] / self.dx[0])

        data = np.array(data)
        row = np.array(row) - self.row_beg
        col = np.array(col)
        self.Apar = sparse.csr_matrix((data, (row, col)), shape=(self.row_end - self.row_beg, self.global_size_A))
        del data, row, col

        # ---- POSTSETUP <end> ----

    # user defined
    def bpar(self, t):
        return self.rhs(t, self.x).flatten()[self.row_beg:self.row_end]

    def u_initial(self, z):
        f = np.empty(self.spatial_points)
        for ix in range(self.spatial_points[0]):
            for iu in range(self.spatial_points[1]):
                for iv in range(self.spatial_points[2]):
                    for iw in range(self.spatial_points[3]):
                        ro = 1 + 0.1 * np.sin(2 * np.pi * self.xx[ix])
                        e = np.exp(-ro * ((self.uu[iu] - 1) ** 2 + self.vv[iv] ** 2 + self.ww[iw] ** 2))
                        f[ix, iu, iv, iw] = ro * np.sqrt(ro / np.pi) ** 3 * e
        return f

    # computing the collision term
    # has to operate on python variables only
    def boltzmann_fft_python(self, f0):
        f_spec = np.fft.fftshift(np.fft.ifftn(f0.astype(complex)))

        # gain term
        f_temp = np.zeros_like(f_spec).astype(complex)
        for ii in range(self.gas_fsm_nm * (self.gas_fsm_nm - 1)):
            fg1 = f_spec * self.phi[:, :, :, ii]
            fg2 = f_spec * self.psi[:, :, :, ii]
            fg11 = np.fft.fftn(fg1)
            fg22 = np.fft.fftn(fg2)
            f_temp += fg11 * fg22

        # loss term
        fl1 = f_spec * self.chi
        fl2 = f_spec.copy()
        fl11 = np.fft.fftn(fl1)
        fl22 = np.fft.fftn(fl2)
        f_temp -= fl11 * fl22

        return 4 * np.pi ** 2 / self.gas_fsm_kn / self.gas_fsm_nm ** 2 * f_temp.real

    # has to operate on python variables only
    def F(self, u):
        Qf = np.empty_like(u, dtype=complex)

        # case with spatial parallelization
        if self.frac > 1:
            Nuvw = self.spatial_points[1] * self.spatial_points[2] * self.spatial_points[3]
            Nx = (self.row_end - self.row_beg) // Nuvw

            if Nx == 0:
                raise RuntimeError('cannot parallelize for Nx < proc_col')

            Q = np.zeros([Nx] + self.spatial_points[1:])
            f = u.reshape([Nx] + self.spatial_points[1:]).real
            for ix in range(Nx):
                Q[ix, :, :, :] = self.boltzmann_fft_python(f[ix, :, :, :])
            Qf = Q.flatten()

        # case without spatial parallelization
        else:
            Q = np.zeros(self.spatial_points)
            for i in range(self.Frac):
                f = u[i * self.global_size_A:(i + 1) * self.global_size_A].reshape(self.spatial_points).real
                for ix in range(self.spatial_points[0]):
                    Q[ix, :, :, :] = self.boltzmann_fft_python(f[ix, :, :, :])
                Qf[i * self.global_size_A:(i + 1) * self.global_size_A] = Q.flatten()
        return Qf

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


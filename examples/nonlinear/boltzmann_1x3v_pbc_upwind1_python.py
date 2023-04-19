import numpy as np
from scipy import sparse
from petsc4py import PETSc
from core.imex_newton_refinement import IMEXNewtonIncrementParalpha
from mpi4py import MPI

# Julia backend
#from julia.api import Julia
#jl = Julia(compiled_modules=False)
from julia import KitBase as kt

"""
boltzman equation
u_t + v2 * u_y = 1 / knudsen * Q(u)
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

    # these HAVE to be python objects
    gas_fsm_kn = None       # = gas.fsm.Kn
    gas_fsm_nm = None       # = gas.fsm.nm
    gas_gamma = None        # = gas.γ
    vs_u = None             # = vs.u
    vs_v = None             # = vs.v
    vs_w = None             # = vs.w
    phi = None
    psi = None
    chi = None

    # PROCESSORS NEEDED = proc_row * proc_col

    def __init__(self):

        super().__init__()

    def setup(self):

        # ---- PRESETUP ----
        # Julia objects
        vs = kt.VSpace3D(self.U_left, self.U_right, self.spatial_points[1], self.V_left, self.V_right, self.spatial_points[2], self.W_left, self.W_right, self.spatial_points[3])
        muref = kt.ref_vhs_vis(self.knudsen, 1.0, 0.5)
        fsm = kt.fsm_kernel(vs, muref, 5, 1.0)
        gas = kt.Gas(Kn=self.knudsen, K=0.0, fsm=fsm)

        # Python numpy.ndarray objects
        phi_, psi_, chi_ = kt.kernel_mode(
            5,
            vs.u1,
            vs.v1,
            vs.w1,
            vs.du[1, 1, 1],
            vs.dv[1, 1, 1],
            vs.dw[1, 1, 1],
            vs.nu,
            vs.nv,
            vs.nw,
            1.0,
        )

        # copy into Python objects
        self.gas_fsm_kn = gas.fsm.Kn
        self.gas_fsm_nm = gas.fsm.nm
        self.gas_gamma = gas.γ
        self.vs_u = vs.u[:, :, :]
        self.vs_v = vs.v[:, :, :]
        self.vs_w = vs.w[:, :, :]
        self.phi = phi_[:, :, :, :]
        self.psi = psi_[:, :, :, :]
        self.chi = chi_[:, :, :]

        self.xx = np.linspace(self.X_left, self.X_right, self.spatial_points[0] + 1)[:-1]

        self.dx = []
        self.dx.append((self.X_right - self.X_left) / self.spatial_points[0])
        self.dx.append(vs.du[1, 1, 1])
        self.dx.append(vs.dv[1, 1, 1])
        self.dx.append(vs.dw[1, 1, 1])

        # x and size_global_A have to be filled before super().setup()
        self.x = np.meshgrid(self.xx, self.vs_u[:, 1, 1], self.vs_v[1, :, 1], self.vs_w[1, 1, :])
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

            if self.vs_u[iu, 1, 1] < 0:
                row.append(i)
                col.append(i)
                data.append(self.vs_u[iu, 1, 1] / self.dx[0])

                row.append(i)
                col.append((i + Nuvw) % self.global_size_A)
                data.append(-self.vs_u[iu, 1, 1] / self.dx[0])

            else:
                row.append(i)
                col.append(i)
                data.append(-self.vs_u[iu, 1, 1] / self.dx[0])

                row.append(i)
                col.append((i - Nuvw) % self.global_size_A)
                data.append(self.vs_u[iu, 1, 1] / self.dx[0])

        data = np.array(data)
        row = np.array(row) - self.row_beg
        col = np.array(col)
        self.Apar = sparse.csr_matrix((data, (row, col)), shape=(self.row_end - self.row_beg, self.global_size_A))
        del data, row, col

        # ---- POSTSETUP <end> ----

    # user defined
    def bpar(self, t):
        return self.rhs(t, self.x).flatten()[self.row_beg:self.row_end]

    # called during setup phase
    def fw(self, x, p):
        ρ = 1 + 0.1 * np.sin(2 * np.pi * x)
        u = 1.0
        λ = ρ
        return kt.prim_conserve([ρ, u, 0, 0, λ], self.gas_gamma)

    # called during setup phase
    def ff(self, x, p):
        w = self.fw(x, p)
        prim = kt.conserve_prim(w, self.gas_gamma)
        return kt.maxwellian(self.vs_u, self.vs_v, self.vs_w, prim)

    # user defined, called during setup phase
    def u_initial(self, z):
        f = np.zeros(self.spatial_points)
        for i in range(self.spatial_points[0]):
            f[i, :, :, :] = self.ff(self.xx[i], None)
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


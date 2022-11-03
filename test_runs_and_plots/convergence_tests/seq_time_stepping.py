import numpy as np
np.set_printoptions(linewidth=np.inf)
import scipy as sp
from pySDC.core.Collocation import CollBase

def Newton(T0, u0, dt, f, df, b, steps, maxiter=10, coll_points=3, restol=1e-6, stol=1e-7):

    no_warnings = True
    coll = CollBase(coll_points, 0, 1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')
    Q = coll.Qmat[1:, 1:]
    t = dt * np.array(coll.nodes)
    D, S = np.linalg.eig(Q)     # S @ D @ Sinv = Q
    Sinv = np.linalg.inv(S)
    spatial_points = u0.shape[0]
    u = np.tile(u0, coll_points)    # initial guess
    r = np.empty_like(u)
    fu = np.empty_like(u)
    ncount = np.zeros(steps)
    res = np.empty(steps)

    Q = sp.sparse.csr_matrix(Q)
    S = sp.sparse.csr_matrix(S)
    Sinv = sp.sparse.csr_matrix(Sinv)

    for k in range(steps):

        # assemble r
        for i in range(coll_points):
            r[i * spatial_points: (i + 1) * spatial_points] = b(T0 + k * dt + t[i])
            fu[i * spatial_points: (i + 1) * spatial_points] = f(u[i * spatial_points: (i + 1) * spatial_points])
        r = np.tile(u[-spatial_points:], coll_points) + dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ r

        tmp = u - dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ fu - r
        res[k] = np.linalg.norm(tmp, np.inf)

        # newton iterations
        while res[k] > restol and ncount[k] < maxiter:

            if res[k] > 1000:
                return u[-spatial_points:], res, ncount
            ncount[k] += 1
            u_old = u.copy()

            # get the Jacobian approximation
            J = 0 * sp.sparse.eye(spatial_points)
            for i in range(coll_points):
                J += 1 / coll_points * df(u[i * spatial_points: (i + 1) * spatial_points])
                fu[i * spatial_points: (i + 1) * spatial_points] = f(u[i * spatial_points: (i + 1) * spatial_points])

            z = dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ (fu - sp.sparse.kron(sp.sparse.eye(coll_points), J) @ u) + r
            z = sp.sparse.kron(Sinv, sp.sparse.eye(spatial_points)) @ z

            # solve systems
            for i in range(coll_points):
                tmp = sp.sparse.eye(spatial_points) - D[i] * dt * J
                z_old = z[i * spatial_points: (i + 1) * spatial_points].copy()
                z[i * spatial_points: (i + 1) * spatial_points], info = sp.sparse.linalg.gmres(tmp, z[i * spatial_points: (i + 1) * spatial_points], tol=stol, atol=0, maxiter=100)
                if info != 0 and no_warnings:
                    print('Warning! Some systems did not finish.')
                    no_warnings = False

            u = sp.sparse.kron(S, sp.sparse.eye(spatial_points)) @ z

            # residual
            tmp = u - dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ fu - r
            res[k] = np.linalg.norm(tmp, np.inf)

    return u[-spatial_points:], res, ncount


def IMEX(T0, u0, dt, F, A, b, steps, maxiter=10, coll_points=3, restol=1e-6, stol=1e-7):

    no_warnings = True
    coll = CollBase(coll_points, 0, 1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')
    Q = coll.Qmat[1:, 1:]
    t = dt * np.array(coll.nodes)
    D, S = np.linalg.eig(Q)     # S @ D @ Sinv = Q
    Sinv = np.linalg.inv(S)
    spatial_points = u0.shape[0]
    u = np.tile(u0, coll_points).astype(complex)    # initial guess
    r = np.empty_like(u)
    ncount = np.zeros(steps)
    res = np.zeros_like(ncount)
    Fu = np.empty(spatial_points * coll_points, dtype=complex)

    Q = sp.sparse.csr_matrix(Q)
    S = sp.sparse.csr_matrix(S)
    Sinv = sp.sparse.csr_matrix(Sinv)

    for k in range(steps):

        # assemble r
        for i in range(coll_points):
            r[i * spatial_points: (i + 1) * spatial_points] = b(T0 + k * dt + t[i])
        r = np.tile(u[-spatial_points:], coll_points) + dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ r

        for i in range(coll_points):
            Fu[i * spatial_points: (i + 1) * spatial_points] = F(u[i * spatial_points: (i + 1) * spatial_points])

        tmp = u - dt * sp.sparse.kron(Q, A) @ u - dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ Fu - r
        res[k] = np.linalg.norm(tmp, np.inf)

        # imex iterations
        while res[k] > restol and ncount[k] < maxiter:
            ncount[k] += 1
            u_old = u.copy()

            z = dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ Fu + r
            z = sp.sparse.kron(Sinv, sp.sparse.eye(spatial_points)) @ z

            # solve systems
            for i in range(coll_points):
                tmp = sp.sparse.eye(spatial_points) - D[i] * dt * A
                z[i * spatial_points: (i + 1) * spatial_points], info = sp.sparse.linalg.gmres(tmp, z[i * spatial_points: (i + 1) * spatial_points], tol=stol, atol=0,  maxiter=50)
                if info != 0 and no_warnings:
                    print('Warning! Some systems did not finish.')
                    no_warnings = False

            u = sp.sparse.kron(S, sp.sparse.eye(spatial_points)) @ z

            for i in range(coll_points):
                Fu[i * spatial_points: (i + 1) * spatial_points] = F(u[i * spatial_points: (i + 1) * spatial_points])

            # residual
            tmp = u - dt * sp.sparse.kron(Q, A) @ u - dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ Fu - r
            res[k] = np.linalg.norm(tmp, np.inf)

    return u[-spatial_points:], res, ncount

# OVO KORISTI
def ParalpHa(T0, u0, dt, F, dF, A, b, steps, alpha, beta=[0], maxiter=10, coll_points=3, restol=1e-6, stol=1e-7, reff_run=None, const=1, m0=1, optimal=False):

    no_warnings = True
    coll = CollBase(coll_points, 0, 1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')
    Q = coll.Qmat[1:, 1:]
    Q = sp.sparse.csr_matrix(Q)
    t = dt * np.array(coll.nodes)
    spatial_points = u0.shape[0]
    u = np.tile(u0, coll_points * steps).reshape(spatial_points * coll_points, steps, order='F') + 0j # initial guess
    z = np.zeros_like(u, dtype=complex)
    r = np.zeros_like(u, dtype=complex)
    bb = np.zeros_like(u, dtype=complex)   # for storing dt (Q x I) b
    Fu = np.zeros(spatial_points * coll_points, dtype=complex)
    ncount = 0
    beta_idx = 0
    alpha_idx = 0
    u_history = []
    res_history = []
    Lip_history = []
    c_history = []
    m_history = []
    M_history = []
    consecutive_iterates = []
    stol_input = stol

    # assemble bb
    for k in range(steps):
        for i in range(coll_points):
            bb[i * spatial_points: (i + 1) * spatial_points, k] = b(T0 + k * dt + t[i])
        bb[:, k] = dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ bb[:, k]

    # compute residual
    if reff_run is not None:
        res = np.linalg.norm(u[-spatial_points:, -1] - reff_run, np.inf)
    else:
        res = 0
        for k in range(steps):
            if k == 0:
                rhs = np.tile(u0, coll_points) + bb[:, k]
            else:
                rhs = np.tile(u[-spatial_points:, k - 1], coll_points) + bb[:, k]
            tmp = u[:, k] - dt * sp.sparse.kron(Q, A) @ u[:, k] - dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ F(u[:, k]) - rhs
            res = max(res, np.linalg.norm(tmp, np.inf))

    res_history.append(res)
    u_history.append(u[-spatial_points:, -1].copy())
    convergence_constant = 0
    consecutive_iterates.append(np.inf)

    # m0 and m1 we do not know
    # m2 we can guess as m2 = L2 / ( 1 - L2 ) |u2-u1| when we get u2, for ncount = 2
    for i in range(2):
        m_history.append(m0)
        M_history.append(m0)

    # Paralha iterations
    while ncount < maxiter:

        #print('k = {}\n======='.format(ncount))

        if ncount >= 1:
            consecutive_iterates.append(np.linalg.norm(u_history[-1] - u_history[-2], np.inf))

        #if ncount >= 3 and optimal and min(m_history[ncount], consecutive_iterates[-1]) < restol:
        #    break
        #elif ncount >= 2 and consecutive_iterates[ncount] < restol:
        #    break

        if ncount >= 2 and res_history[-1] < restol:
            break

        if res_history[-1] / res_history[0] > 10000:
            # ncount = 0
            # res_history = []
            break

        # assemble Javg
        J = 0 * sp.sparse.eye(spatial_points)
        if beta[beta_idx] != 0:
            for k in range(steps):
                J += 1 / steps * dF(u[-spatial_points:, k])

        # assemble r
        r[:, :] = bb
        for k in range(steps):
            for i in range(coll_points):
                Fu[i * spatial_points: (i + 1) * A.shape[0]] = F(u[i * spatial_points: (i + 1) * spatial_points, k])
            r[:, k] += dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ (Fu - beta[beta_idx] * sp.sparse.kron(sp.sparse.eye(coll_points), J) @ u[:, k])

        delta_parameter = np.linalg.norm(r, np.inf)

        for k in range(1, steps, 1):
                r[:, k] += (1 - const) * np.tile(u[-spatial_points:, k-1], coll_points)
                r[:, k] *= alpha[alpha_idx] ** (k / steps)
        r[:, 0] += np.tile(u0, coll_points) - alpha[alpha_idx] * np.tile(u[-spatial_points:, -1], coll_points)

        # do fft
        z *= 0
        for j in range(steps):
            for k in range(steps):
                z[:, j] += np.exp(-2 * np.pi * 1j * j * k / steps) * r[:, k]

        # solve decoupled systems
        for k in range(steps):
            G = np.zeros((coll_points, coll_points), dtype=complex)
            G[:, -1] = -const * alpha[alpha_idx] ** (1 / steps) * np.exp(-2 * np.pi * 1j * k / steps)
            G += np.eye(coll_points)
            Ginv = np.linalg.inv(G)
            D, S = np.linalg.eig(Q @ Ginv)  # S @ D @ Sinv = Q @ Ginv
            Sinv = np.linalg.inv(S)

            S = sp.sparse.csr_matrix(S)
            Sinv = sp.sparse.csr_matrix(Sinv)

            # solve the inner system via diagonalization in 3 steps
            r[:, k] = sp.sparse.kron(Sinv, sp.sparse.eye(spatial_points)) @ z[:, k]
            for i in range(coll_points):
                tmp = sp.sparse.eye(spatial_points) - D[i] * dt * (A + beta[beta_idx] * J)
                z[i * spatial_points:(i + 1) * spatial_points, k], info = sp.sparse.linalg.gmres(tmp, r[i * spatial_points:(i + 1) * spatial_points, k], tol=stol, atol=0, maxiter=100, x0=np.zeros(spatial_points))
                if info != 0 and no_warnings:
                    print('Warning! Some systems did not finish.', info)
                    no_warnings = False
            r[:, k] = sp.sparse.kron(S, sp.sparse.eye(spatial_points)) @ z[:, k]
            z[:, k] = sp.sparse.kron(Ginv, sp.sparse.eye(spatial_points)) @ r[:, k]

        # do ifft
        r *= 0
        for j in range(steps):
            for k in range(steps):
                r[:, j] += np.exp(2 * np.pi * 1j * j * k / steps) * z[:, k]

        # scale
        for k in range(steps):
            u[:, k] = alpha[alpha_idx] ** (-k / steps) / steps * r[:, k]

        # compute residual
        if reff_run is not None:
            res = np.linalg.norm(u[-spatial_points:, -1] - reff_run, np.inf)
        else:
            res = 0
            for k in range(steps):
                if k == 0:
                    rhs = np.tile(u0, coll_points) + bb[:, k]
                else:
                    rhs = np.tile(u[-spatial_points:, k - 1], coll_points) + bb[:, k]
                tmp = u[:, k] - dt * sp.sparse.kron(Q, A) @ u[:, k] - dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ F(u[:, k]) - rhs
                res = max(res, np.linalg.norm(tmp, np.inf))

        res_history.append(res)
        u_history.append(u[-spatial_points:, -1].copy())
        #print('new iteration computed')
        ncount += 1

        # compute alphas, m, M
        if ncount >= 2 and optimal:
            Lip = np.linalg.norm(u_history[-1] - u_history[-2], np.inf) / np.linalg.norm(u_history[-2] - u_history[-3], np.inf) # this is a lower approximation of the real Lip. constant

            if Lip < 1:
                # approximate an upper bound for m2
                if ncount == 2:
                    m0 = Lip ** (ncount - 1) / (1 - Lip) * np.linalg.norm(u_history[-1] - u_history[-2], np.inf)
                    m_history.append(m0)

            else:
                print('==========================================================================')
                print('                             WARNING: Lip > 1                             ')
                print('==========================================================================')

            convergence_constant = (1 - alpha[alpha_idx]) * Lip - alpha[alpha_idx]
            eps = np.finfo(float).eps

            #stol = max(eps, 0.1 * abs(restol * (convergence_constant - 1) ** 2 / (4 * delta_parameter * steps) - 3 * eps))
            #stol = min(stol, stol_input)
            #print('stol = ', stol)

            print('convergence limit: ', 4 * steps * (3 * eps + stol) * delta_parameter / (convergence_constant - 1) ** 2)

            print('Lip = {}'.format(Lip))
            print('c = {}'.format(convergence_constant))
            print('res = {} <= m[{}] = {} ? {}'.format(res_history[-1], ncount, m_history[-1], m_history[-1] >= res_history[-1]))

            gamma_parameter = steps * (3 * eps + stol)
            new_alpha = np.sqrt(gamma_parameter * delta_parameter / m0)
            alpha.append(new_alpha)
            m0 = convergence_constant * m0 + 2 * np.sqrt(gamma_parameter * delta_parameter * m0) + gamma_parameter * np.linalg.norm(u[-coll_points * spatial_points:, -1], np.inf)

            m_history.append(m0)
            Lip_history.append(Lip)
            c_history.append(convergence_constant)

        #print('alpha[{}] = {}'.format(ncount, alpha[alpha_idx]))
        #print('\n')
        alpha_idx = min(len(alpha) - 1, alpha_idx + 1)
        beta_idx = min(len(beta) - 1, beta_idx + 1)

    return u[-spatial_points:, -1], res, ncount, u_history, res_history, c_history, Lip_history, alpha, m_history, M_history, consecutive_iterates

def Parallel_IMEX_refinement(T0, u0, dt, F, dF, A, b, steps, alpha, beta=[0], maxiter=10, coll_points=3, restol=1e-6, stol=1e-7, reff_run=None, const=1, m0=1, optimal=False):

    no_warnings = True
    coll = CollBase(coll_points, 0, 1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')
    Q = coll.Qmat[1:, 1:]
    Q = sp.sparse.csr_matrix(Q)
    t = dt * np.array(coll.nodes)
    spatial_points = u0.shape[0]
    u = np.tile(u0, coll_points * steps).reshape(spatial_points * coll_points, steps, order='F') + 0j # initial guess
    z = np.zeros_like(u, dtype=complex)
    r = np.zeros_like(u, dtype=complex)
    bb = np.zeros_like(u, dtype=complex)   # for storing dt (Q x I) b
    res = np.zeros_like(u, dtype=complex)
    Fu = np.zeros(spatial_points * coll_points, dtype=complex)
    ncount = 0
    beta_idx = 0
    alpha_idx = 0
    u_history = []
    res_history = []
    err_history = []
    consecutive_iterates = []

    # assemble bb
    for k in range(steps):
        for i in range(coll_points):
            bb[i * spatial_points: (i + 1) * spatial_points, k] = b(T0 + k * dt + t[i])
        bb[:, k] = dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ bb[:, k]

    # compute error
    if reff_run is not None:
        err = np.linalg.norm(u[-spatial_points:, -1] - reff_run, np.inf)

    # compute residual
    for k in range(steps):
        if k == 0:
            rhs = np.tile(u0, coll_points) + bb[:, k]
        else:
            rhs = np.tile(u[-spatial_points:, k - 1], coll_points) + bb[:, k]
        res[:, k] = u[:, k] - dt * sp.sparse.kron(Q, A) @ u[:, k] - dt * sp.sparse.kron(Q, sp.sparse.eye(spatial_points)) @ F(u[:, k]) - rhs

    res_history.append(np.linalg.norm(res.flatten(), np.inf))
    err_history.append(err)
    u_history.append(u[-spatial_points:, -1].copy())
    consecutive_iterates.append(np.inf)

    # Paralha iterations
    while ncount < maxiter:

        #print('k = {}\n======='.format(ncount))

        if ncount >= 1:
            consecutive_iterates.append(np.linalg.norm(u_history[-1] - u_history[-2], np.inf))

        if err_history[-1] < restol:
            break

        if err_history[-1] / err_history[0] > 10000:
            break

        # assemble Javg
        J = 0 * sp.sparse.eye(spatial_points)
        if beta[beta_idx] != 0:
            for k in range(steps):
                J += 1 / steps * dF(u[-spatial_points:, k])

        # scale with J
        for k in range(1, steps, 1):
                res[:, k] += (1 - const) * np.tile(u[-spatial_points:, k-1], coll_points)
                res[:, k] *= alpha[alpha_idx] ** (k / steps)

        # do fft
        z *= 0
        for j in range(steps):
            for k in range(steps):
                z[:, j] += np.exp(-2 * np.pi * 1j * j * k / steps) * res[:, k]

        # solve decoupled systems
        for k in range(steps):
            G = np.zeros((coll_points, coll_points), dtype=complex)
            G[:, -1] = -const * alpha[alpha_idx] ** (1 / steps) * np.exp(-2 * np.pi * 1j * k / steps)
            G += np.eye(coll_points)
            Ginv = np.linalg.inv(G)
            D, S = np.linalg.eig(Q @ Ginv)  # S @ D @ Sinv = Q @ Ginv
            Sinv = np.linalg.inv(S)

            S = sp.sparse.csr_matrix(S)
            Sinv = sp.sparse.csr_matrix(Sinv)

            # solve the inner system via diagonalization in 3 steps
            r[:, k] = sp.sparse.kron(Sinv, sp.sparse.eye(spatial_points)) @ z[:, k]
            for i in range(coll_points):
                tmp = sp.sparse.eye(spatial_points) - D[i] * dt * (A + beta[beta_idx] * J)
                z[i * spatial_points:(i + 1) * spatial_points, k], info = sp.sparse.linalg.gmres(tmp, r[i * spatial_points:(i + 1) * spatial_points, k], tol=stol, atol=0, maxiter=100, x0=np.zeros(spatial_points))
                if info != 0 and no_warnings:
                    print('Warning! Some systems did not finish.', info)
                    no_warnings = False
            r[:, k] = sp.sparse.kron(S, sp.sparse.eye(spatial_points)) @ z[:, k]
            z[:, k] = sp.sparse.kron(Ginv, sp.sparse.eye(spatial_points)) @ r[:, k]

        # do ifft
        r *= 0
        for j in range(steps):
            for k in range(steps):
                r[:, j] += np.exp(2 * np.pi * 1j * j * k / steps) * z[:, k]

        # scale
        for k in range(steps):
            r[:, k] = alpha[alpha_idx] ** (-k / steps) / steps * r[:, k]

        # update
        u += r

        # compute residual
        if reff_run is not None:
            err = np.linalg.norm(u[-spatial_points:, -1] - reff_run, np.inf)

        # compute residual
        for k in range(steps):
            if k == 0:
                rhs = np.tile(u0, coll_points) + bb[:, k]
            else:
                rhs = np.tile(u[-spatial_points:, k - 1], coll_points) + bb[:, k]
            res[:, k] = u[:, k] - dt * sp.sparse.kron(Q, A) @ u[:, k] - dt * sp.sparse.kron(Q, sp.sparse.eye(
                spatial_points)) @ F(u[:, k]) - rhs
        res_history.append(np.linalg.norm(res.flatten(), np.inf))

        err_history.append(err)
        u_history.append(u[-spatial_points:, -1].copy())
        #print('new iteration computed')
        ncount += 1
        alpha_idx = min(len(alpha) - 1, alpha_idx + 1)
        beta_idx = min(len(beta) - 1, beta_idx + 1)

    return u[-spatial_points:, -1], u_history, err_history, res_history, consecutive_iterates
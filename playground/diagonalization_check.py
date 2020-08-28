import numpy as np
np.set_printoptions(precision=2, linewidth=np.inf)

L = 400
a = 0.000054

E = -np.eye(L, k=-1, dtype=complex)
E[0, -1] = -a
J = np.eye(L, dtype=complex)
Jinv = np.eye(L, dtype=complex)
D = np.eye(L, dtype=complex)
for l in range(L):
    J[l, l] = a**(-l/L)
    Jinv[l, l] = a**(l/L)
    D[l, l] = -a**(1/L) * np.exp(-2 * np.pi * 1j * l/L)

w = np.exp(2 * np.pi * 1j / L)
F = np.empty((L, L), dtype=complex)
Fo = np.empty((L, L), dtype=complex)
for j in range(L):
    for k in range(L):
        F[j, k] = w**(j*k)
        Fo[k, j] = np.conj(F[j, k])
V = J @ F
Vinv = 1/L * Fo @ Jinv

Err = E - V @ D @ Vinv
#print(Err)
err = np.linalg.norm(Err, 'fro')
print(err)




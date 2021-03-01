import numpy as np
import sys
import matplotlib.pyplot as plt
# 15/4	−77/6	107/6	−13     61/12	−5/6
np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)
sigma = 1
p = 1
# A @ f = -i * ddf

def f(z, t):
    a = sigma + 2j * t
    return 1/a * np.exp(0.5 * (1/a - 1) + 1j/a * z - 0.5 /a * z**2)
    # return np.exp(-z)

def ddf(z, t):
    a = sigma + 2j * t
    exp = np.exp(0.5 * (1/a - 1) + 1j/a * z - 0.5 /a * z**2)
    ut = -1j/a**2 * exp + 1/a * exp * (-1j/a**2 + 2/a**2 * z + 1j/a**2 * z**2)

    return -1j * ut
    # return -1/a**2 * exp
    # return np.exp(-z)

N = 24000
A = np.zeros((N, N), dtype=complex)
# forward laplace 4
A += 15/4 * np.eye(N) - 77/6 * np.eye(N, N, 1) + 107/6 * np.eye(N, N, 2) - 13 * np.eye(N, N, 3) + 61/12 * np.eye(N, N, 4) - 5/6 * np.eye(N, N, 5)
# A += - 77/6 * np.eye(N, N, -(N-1)) + 107/6 * np.eye(N, N, -(N-2)) - 13 * np.eye(N, N, -(N-3)) + 61/12 * np.eye(N, N, -(N-4)) - 5/6 * np.eye(N, N, -(N-5))

xvals = np.linspace(-10, 10, N)
T = 0.1
dx = 20 / (N-1)
A /= dx**2

res = A @ f(xvals, T) - ddf(xvals, T)
err = np.linalg.norm(res, np.inf)
print(err)

plt.plot(res.imag)
plt.plot(res.real)
plt.show()


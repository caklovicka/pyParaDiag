import numpy as np
import matplotlib.pyplot as plt
# 15/4	−77/6	107/6	−13     61/12	−5/6
np.set_printoptions(linewidth=np.inf)
sigma = 1
p = 1
def f(z, t):
    a = sigma + 2j * t
    return 1/a * np.exp(0.5 * (1/a - 1) + 1j/a * z - 0.5 /a * z**2)

def ddf(z, t):
    a = sigma + 2j * t
    exp = np.exp(0.5 * (1/a - 1) + 1j/a * z - 0.5 /a * z**2)

    return -2j/a**2 * exp + 1/a * exp * (-1j/a**2 + 2/a**2 * z + 1j/a**2 * z**2)

N = 1000
A = np.zeros((N, N), dtype=complex)
# forward laplace 4
A += 15/4 * np.eye(N) - 77/6 * np.eye(N, N, 1) + 107/6 * np.eye(N, N, 2) - 13 * np.eye(N, N, 3) + 61/12 * np.eye(N, N, 4) - 5/6 * np.eye(N, N, 5)
# A += - 77/6 * np.eye(N, N, -(N-1)) + 107/6 * np.eye(N, N, -(N-2)) - 13 * np.eye(N, N, -(N-3)) + 61/12 * np.eye(N, N, -(N-4)) - 5/6 * np.eye(N, N, -(N-5))

xvals = np.linspace(-6, 6, N)
T = 0
dx = (xvals[-1] - xvals[0])/(N-1)
dx = xvals[1] - xvals[0]
A /= dx**2

res = 1j * A @ f(xvals, T) - ddf(xvals, T)
err = np.linalg.norm(res, np.inf)
print(err)

plt.plot(res.imag)
plt.plot(res.real)
plt.show()


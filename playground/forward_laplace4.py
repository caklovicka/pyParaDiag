import numpy as np
import matplotlib.pyplot as plt
# 15/4	−77/6	107/6	−13     61/12	−5/6
np.set_printoptions(linewidth=np.inf)

def f(x):
    return np.sin(x)

def ddf(x):
    return -np.sin(x)

N = 100
A = 15/4 * np.eye(N) - 77/6 * np.eye(N, N, 1) + 107/6 * np.eye(N, N, 2) - 13 * np.eye(N, N, 3) + 61/12 * np.eye(N, N, 4) - 5/6 * np.eye(N, N, 5)
# make circulant
A += - 77/6 * np.eye(N, N, -(N-1)) + 107/6 * np.eye(N, N, -(N-2)) - 13 * np.eye(N, N, -(N-3)) + 61/12 * np.eye(N, N, -(N-4)) - 5/6 * np.eye(N, N, -(N-5))
dx = 1/(N-1)
A /= dx

xvals = np.linspace(0, 2 * np.pi, N)

res = A @ f(xvals) - ddf(xvals)
err = np.linalg.norm(res, np.inf)
print(err)

plt.plot(res)
plt.show()


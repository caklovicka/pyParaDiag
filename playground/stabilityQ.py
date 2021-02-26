from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
import numpy as np
import matplotlib.pyplot as plt

dt = 1
M = 2
N = 60


coll = CollGaussRadau_Right(num_nodes=M, tleft=0, tright=1)
t = dt * np.array(coll.nodes)
Q = coll.Qmat[1:, 1:]
eigQ = np.array(np.linalg.eigvals(Q))

lambda_real = np.linspace(-1, 1, N)
lambda_complex = np.linspace(2.3, 3.5, N)

for lr in lambda_real:
    for lc in lambda_complex:
        ro = min(abs(1 - (lr + lc * 1j) * eigQ))
        if ro < 1:
            plt.plot(lr, lc, 'r.')
        elif ro > 1:
            plt.plot(lr, lc, 'g.')
        else:
            plt.plot(lr, lc, 'b.')

# plt.plot(lambda_real, 0 * lambda_real, 'k-')
plt.plot(0 * lambda_complex, lambda_complex, 'k-')

plt.show()
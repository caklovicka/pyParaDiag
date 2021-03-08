import numpy as np
import sys
np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize, precision=5)
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right

M = 3
alpha = 0.1
L = 64
dt = 1

d = -alpha**(1/L)
r = d/(1+d)

# get eigens of Q
coll = CollGaussRadau_Right(num_nodes=M, tleft=0, tright=1)
t = np.array(coll.nodes)
Q = dt * coll.Qmat[1:, 1:]

rDH = np.zeros_like(Q)
H = np.zeros_like(Q)
for i in range(M):
    rDH[i, -1] = r * t[i]
    H[i, -1] = 1

QGinv = Q - rDH
G = np.eye(M) + d * H
Ginv = np.eye(M) - r * H

# G * Ginv x = x
x = np.random.random(M) * 0.1
y = G @ x
z = Ginv @ y

print(np.linalg.norm(z - x, 2), np.linalg.norm(x, 2))


# print(np.linalg.cond(np.eye(M) - QGinv), np.linalg.cond(G), np.linalg.cond(Ginv), r, d)
# print(np.linalg.norm(G @ Ginv - np.eye(M)))





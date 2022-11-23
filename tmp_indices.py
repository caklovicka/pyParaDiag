import numpy as np

n = [10, 13, 6, 23]
a = np.random.rand(n[0], n[1], n[2], n[3])
af = a.flatten()

n123 = n[1] * n[2] * n[3]
n23 = n[2] * n[3]
n3 = n[3]

for i in range(af.shape[0]):
    i0 = i // n123
    i1 = i % n123 // n23
    i2 = i % n23 // n3
    i3 = i % n3

    if np.abs(af[i] - a[i0, i1, i2, i3]) > 1e-10:
        print(af[i] - a[i0, i1, i2, i3])
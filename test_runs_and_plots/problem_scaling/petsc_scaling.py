import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


path = ['adv1_strong/output/000002/result/result.dat']

# nproc | tot_time |
eq = np.loadtxt(path[0], delimiter='|', usecols=[0, 3], skiprows=3)
n = eq.shape[0]
seqT = max(eq[:, 1])
print(eq)
eq[:, 1] = seqT / eq[:, 1]

plt.plot(np.log2(eq[:, 0]), eq[:, 1])
plt.plot(range(n-3), 2**np.array(range(n-3)))
plt.xticks(range(8), eq[:, 0])
plt.legend(['petsc-juwels', 'ideal'])
plt.show()
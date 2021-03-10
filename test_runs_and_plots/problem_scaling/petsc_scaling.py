import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


path = ['adv1_strong/output/000004/result/result.dat']

# nproc | tot_time |
eq = np.loadtxt(path[0], delimiter='|', usecols=[0, 3], skiprows=3)
n = eq.shape[0]
seqT = max(eq[:, 1])
print(eq)
eq[:, 1] = seqT / eq[:, 1]

plt.plot(np.log2(eq[:, 0]), eq[:, 1])
plt.plot(np.log2(eq[:-3, 0]), eq[:-3, 0])
plt.xticks(np.log2(eq[:, 0]), eq[:, 0])
plt.legend(['petsc-juwels', 'ideal'])
plt.show()
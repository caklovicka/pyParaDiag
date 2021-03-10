import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


path = ['adv3_strong/output/000002/result/result.dat']

# nproc | tot_time |
eq = np.loadtxt(path[0], delimiter='|', usecols=[0, 1], skiprows=3)
seqT = max(eq[:, 1])
print(eq)
eq[:, 1] = seqT / eq[:, 1]


plt.plot(np.log2(eq[:, 0]), eq[:, 1])
plt.plot(np.log2(eq[:, 0]), [1, 2, 4, 8, 16, 32, 64])
plt.xticks(np.log2(eq[:, 0]), eq[:, 0])
plt.show()
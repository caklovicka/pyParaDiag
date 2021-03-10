import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


path = ['adv1_strong/output/000006/result/result.dat', 'adv1_strong/output/000007/result/result.dat']

# nproc | tot_time |
eq = []
for p in path:
    eq.append(np.loadtxt(p, delimiter='|', usecols=[0, 3], skiprows=3))

n = eq[0].shape[0]
seqT = 10.3642304
for i in range(len(eq)):
    eq[i][:, 1] = seqT / eq[i][:, 1]

    plt.plot(np.log2(eq[i][:, 0]), eq[i][:, 1], 'X')
    # plt.plot(np.log2(eq[:, 0]), eq[:, 0], 'x')
    plt.xticks(np.log2(eq[i][:, 0]), eq[i][:, 0])

plt.legend(['petsc-6', 'petsc-12', 'ideal'])
# plt.xlim([0, 5])
# plt.ylim([0, 15])
plt.show()
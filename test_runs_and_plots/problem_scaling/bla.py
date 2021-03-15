import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


petsc_run = np.loadtxt('adv1_strong/output/000005/result/result.dat', delimiter='|', usecols=[0, 3], skiprows=3)
petsc_run2 = np.loadtxt('adv1_strong/output/000003/result/result.dat', delimiter='|', usecols=[0, 3], skiprows=3)
print(petsc_run2)

seqT = 10.364230

indices = np.argsort(petsc_run[:, 0])
petsc_run = petsc_run[indices, :]
petsc_run[:, 1] = seqT / petsc_run[:, 1]

plt.plot(np.log2(petsc_run[:, 0]), petsc_run[:, 1])
ticks = list(np.log2(petsc_run[:, 0]))
labels = list(petsc_run[:, 0])

petsc_run = petsc_run2.copy()
indices = np.argsort(petsc_run[:, 0])
petsc_run = petsc_run[indices, :]
petsc_run[:, 1] = seqT / petsc_run[:, 1]

plt.plot(np.log2(petsc_run[:, 0]), petsc_run[:, 1])
ticks += list(np.log2(petsc_run[:, 0]))
labels += list(petsc_run[:, 0])

plt.xticks(ticks, labels)
plt.grid(True)
plt.legend(['48', '24'])
# plt.xlim([0, 5])
# plt.ylim([0, 15])
plt.show()
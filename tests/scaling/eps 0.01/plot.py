import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# beta | nproc | time | convergence
table = np.loadtxt('output1/000000/result/result.dat', delimiter='|', skiprows=3, usecols=[1, 2, 5, 12])

imex_proc = []
imex_time = []

newton_proc = []
newton_time = []

for i in range(table.shape[0]):
    if table[i, 3] == 0:
        continue
    if table[i, 0] == 0:
        imex_proc.append(table[i, 1])
        imex_time.append(table[i, 2])
    elif table[i, 0] == 1:
        newton_proc.append(table[i, 1])
        newton_time.append(table[i, 2])

plt.semilogx(imex_proc, imex_time, 'X-')
plt.semilogx(newton_proc, newton_time, 'X--')
plt.legend(['imex', 'newton'])
plt.xlabel('cores')
plt.ylabel('time[s]')
plt.xticks([1, 2, 4, 8, 16, 64], [1, 2, 4, 8, 16, 64])
plt.show()
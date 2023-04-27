import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

legend = []
petsc_proc = []
petsc_time = []
custom_lines = []
mksz = 10
lw = 2
plt.figure(figsize=(6, 5))
col = sns.color_palette("bright", 3)

runs = ['1e-1', '5e-2', '1e-2']
file = 'data/boltzmann2_petsc32_pint_k={}.dat'
K = len(runs)

pint_petsc_proc = []
pint_petsc_time = []
pint_petsc_iters = []

for k in range(K):
    # nproc | time
    table = np.loadtxt(file.format(runs[k]), delimiter='|', skiprows=3, usecols=[1, 5])

    pint_petsc_proc.append([])
    pint_petsc_time.append([])

    legend.append(r'$\varepsilon$ = {}'.format(runs[k]))
    custom_lines.append(Line2D([0], [0], color=col[k], linestyle='--'))

    for i in range(table.shape[0]):
        pint_petsc_proc[k].append(np.log2(table[i, 0]))
        pint_petsc_time[k].append(table[i, 1])

    for i in range(len(pint_petsc_time[k])):
        plt.semilogy(pint_petsc_proc[k][i], pint_petsc_time[k][i], color=col[k], marker='X', markersize=mksz)

    plt.semilogy(pint_petsc_proc[k], pint_petsc_time[k], linestyle='--', color=col[k], linewidth=lw)

plt.legend(custom_lines, legend)
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('wall clock time [s]', fontsize=12)
plt.xticks([5, 6, 7, 8, 9, 10], [32, 64, 128, 256, 512, 1024])
plt.grid('gray')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

legend = []
petsc_proc = []
petsc_time = []
custom_lines = []
mksz = 20
lw = 2
plt.figure(figsize=(12, 6))

# just PETSc
# nproc | time
table = np.loadtxt('output1/000000/result/result.dat', delimiter='|', skiprows=3, usecols=[1, 2])
seq_time = table[0, 1]

for i in range(table.shape[0]):
    petsc_proc.append(table[i, 0])
    petsc_time.append(table[i, 1])
    if petsc_proc[-1] == 1:
        petsc_time_seq = petsc_time[-1]

all_proc = petsc_proc + [64, 128, 256, 512, 1024, 2048]

s = petsc_time_seq/petsc_time
e = petsc_time_seq/(np.array(petsc_time) * np.array(petsc_proc))

plt.subplot(121)
plt.semilogx(petsc_proc, np.log2(petsc_time_seq/np.log10(petsc_time)), 'X-', color='gray', markersize=mksz // 2, linewidth=lw)

plt.subplot(122)
plt.semilogx(petsc_proc, petsc_time_seq/(np.array(petsc_time) * np.array(petsc_proc)), 'X-', color='gray', markersize=mksz // 2, linewidth=lw)

# PinT + PETSc
runs = [3, 4, 5]
K = len(runs)
runs = [3, 4, 5]
col = sns.color_palette("bright", K)

pint_petsc_proc = []
pint_petsc_time = []
pint_petsc_iters = []

custom_lines.append(Line2D([0], [0], color='gray', linestyle='-'))
legend.append('petsc')

for k in range(K):
    # nproc | proc_col | time | tot iters
    table = np.loadtxt('output1/00000{}/result/result.dat'.format(runs[k]), delimiter='|', skiprows=3, usecols=[1, 2, 5, 8])

    pint_petsc_proc.append([])
    pint_petsc_time.append([])
    pint_petsc_iters.append([])

    proc_col = table[0, 1]
    legend.append('petsc({}) + pint '.format(int(proc_col)))
    custom_lines.append(Line2D([0], [0], color=col[k], linestyle='--'))

    pint_petsc_proc[k].append(int(proc_col))
    pint_petsc_time[k].append(petsc_time[-(k+1)])

    for i in range(table.shape[0]):
        pint_petsc_proc[k].append(table[i, 0])
        pint_petsc_time[k].append(table[i, 2])

        rolling = 32 / (table[i, 0] / table[i, 1])

plt.subplot(121)
for k in range(K):
    plt.semilogx(pint_petsc_proc[k], np.log2(petsc_time_seq/np.log10(pint_petsc_time[k])), 'X--', color=col[k], linewidth=lw)

plt.legend(custom_lines, legend)
plt.xlabel('total number of cores')
plt.ylabel('log2(speedup)')
plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
plt.grid('gray')

plt.subplot(122)
for k in range(K):
    plt.semilogx(pint_petsc_proc[k], petsc_time_seq/(np.array(pint_petsc_time[k]) * np.array(pint_petsc_proc[k])), 'X--', color=col[k], linewidth=lw)

plt.legend(custom_lines, legend)
plt.xlabel('total number of cores')
plt.ylabel('efficiency')
plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
plt.grid('gray')

plt.tight_layout()
plt.show()

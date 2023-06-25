import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mksz = 8
lw = 4
col = sns.color_palette("hls", 9)

file1 = ['data/ac001_1.dat', 'data/ac001_2.dat', 'data/ac001_3.dat']
file2 = ['data/ac1_0.dat', 'data/ac2_2.dat', 'data/ac3_2.dat']

legend = [r'$\varepsilon^2 = 1, M=1$', r'$\varepsilon^2 = 1, M=2$', r'$\varepsilon^2 = 1, M=3$',
          r'$\varepsilon^2 = 0.0001, M=1$', r'$\varepsilon^2 = 0.0001, M=2$', r'$\varepsilon^2 = 0.0001, M=3$',
          r'$\varepsilon^2 = 0.2, M=1$ (PFASST)', r'$\varepsilon^2 = 0.2, M=1$ (PFASST)', r'$\varepsilon^2 = 0.2, M=1$ (PFASST)']

newton_proc = []
newton_time = []
plt.figure(figsize=(9, 6), dpi=150)

for k in range(len(file1)):
    # beta | nproc | time | tot iters | convergence | diff
    table = np.loadtxt(file1[k], delimiter='|', skiprows=3, usecols=[1, 2, 5, 8, 12, 13])

    newton_proc.append([])
    newton_time.append([])

    for i in range(table.shape[0]):

        if table[i, 4] == 0 or table[i, 5] > 1e-3:
            continue

        rolling = 64 / table[i, 1]

        if table[i, 0] == 1:
            newton_proc[k].append(np.log2(table[i, 1]))
            newton_time[k].append(table[i, 2])

for k in range(3):
    # beta | nproc | rolling | time | tot iters | convergence
    table = np.loadtxt(file2[k], delimiter='|', skiprows=3, usecols=[1, 2, 3, 5, 8, 12])

    newton_proc.append([])
    newton_time.append([])

    for i in range(table.shape[0]):

        if table[i, 5] == 0:
            continue

        rolling = table[i, 2]

        if table[i, 0] == 1:
            newton_proc[-1].append(np.log2(table[i, 1]))
            newton_time[-1].append(table[i, 3])

# PFASST
# M = 1
newton_time.append([1.51, 1.16, 0.67, 0.4, 0.27, 0.18, 0.14, 0.13, 0.15, 0.16, 0.20])
newton_proc.append(list(np.log2(np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]))))

# M = 2
newton_time.append([0.54, 0.42, 0.25, 0.16, 0.11, 0.07, 0.07, 0.07, 0.08])
newton_proc.append(list(np.log2(np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]))))

# M = 3
newton_time.append([0.23, 0.20])
newton_proc.append(list(np.log2(np.array([1, 2]))))

for k in range(6):
    plt.semilogy(newton_proc[k], newton_time[k], '--X', color=col[k], markersize=mksz)
for k in range(6, 9, 1):
    plt.semilogy(newton_proc[k], newton_time[k], ':X', color=col[k], markersize=mksz)

plt.legend(legend, bbox_to_anchor=(1.1, 1.05))
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('wallclock time [s]', fontsize=12)
plt.grid('gray')

xx = []
yy = []
for k in range(9):
    xx += newton_proc[k]

for x in xx:
    yy.append(int(2 ** x))
plt.xticks(xx, yy)

plt.tight_layout()
plt.show()
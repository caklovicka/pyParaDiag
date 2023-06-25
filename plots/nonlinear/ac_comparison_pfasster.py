import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mksz = 12
lw = 2
col = sns.color_palette("hls", 2)

run = 1

if run == 0:
    file = ['data/ac4_24.dat']
    legend = ['parallel across steps']
else:
    file = ['data/ac4_24_p.dat']
    legend = ['parallel across steps and nodes']

K = len(file)

newton_proc = []
newton_time = []
plt.figure(figsize=(5, 4), dpi=150)

for k in range(K):
    # nproc | time
    table = np.loadtxt(file[k], delimiter='|', skiprows=3, usecols=[1, 4])
    print(table)

    newton_proc.append([])
    newton_time.append([])

    for i in range(table.shape[0]):

        rolling = 64 / table[i, 1]

        newton_proc[k].append(np.log2(table[i, 0]))
        newton_time[k].append(table[i, 1])

    plt.semilogy(newton_proc[k], newton_time[k], '-X', color=col[k], linewidth=lw, markersize=mksz)

plt.legend(legend)
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('wallclock time [s]', fontsize=12)
plt.grid('gray')

if run == 0:
    plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 4, 8, 12, 24])
    plt.yticks([300, 250, 200, 150, 100, 50], [300, 250, 200, 150, 100, 50])
else:
    plt.xticks([2, 3, 4, 5, 6, 7], [4, 8, 12, 24, 48, 96])
    plt.yticks([100, 90, 80, 70, 60, 50, 40, 30, 20], [100, 90, 80, 70, 60, 50, 40, 30, 20])

plt.xlim([1.5, 7.5])
plt.tight_layout()
plt.show()
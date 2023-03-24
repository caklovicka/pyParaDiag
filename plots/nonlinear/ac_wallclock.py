import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mksz = 16
lw = 2
col = sns.color_palette("hls", 3)


#file = ['data/ac001_1.dat', 'data/ac001_2.dat', 'data/ac001_3.dat']
file = ['data/ac_small_dt.dat']
K = len(file)

legend = ['1e-5 (imex)', '1e-5 (newton)', '1e-9 (imex)', '1e-9 (newton)', '1e-12 (imex)', '1e-12 (newton)']
if K == 1:
    legend = ['1e-5 (imex)']

imex_proc = []
imex_time = []
imex_its = []
newton_proc = []
newton_time = []
newton_its = []
plt.figure(figsize=(5, 4), dpi=150)

for k in range(K):
    # beta | nproc | time | tot iters | convergence | diff
    table = np.loadtxt(file[k], delimiter='|', skiprows=3, usecols=[1, 2, 5, 8, 12, 13])

    #legend.append('imex ' + str(k + 1))
    #legend.append('newton ' + str(k + 1))

    imex_proc.append([])
    imex_time.append([])
    imex_its.append([])
    newton_proc.append([])
    newton_time.append([])
    newton_its.append([])

    for i in range(table.shape[0]):

        if table[i, 4] == 0 or table[i, 5] > 1e-3:
            continue

        rolling = 64 / table[i, 1]
        # rolling = 1

        if table[i, 0] == 0:
            imex_proc[k].append(np.log2(table[i, 1]))
            imex_time[k].append(table[i, 2])
            imex_its[k].append('$' + str(round(table[i, 3] / rolling)) + '$')

        elif table[i, 0] == 1:
            newton_proc[k].append(np.log2(table[i, 1]))
            newton_time[k].append(table[i, 2])
            newton_its[k].append('$' + str(round(table[i, 3] / rolling)) + '$')

    plt.semilogy(imex_proc[k], imex_time[k], ':', color=col[k], linewidth=lw)
    plt.semilogy(newton_proc[k], newton_time[k], '-', color=col[k], linewidth=lw)

for k in range(K):
    for i in range(len(imex_its[k])):
        plt.semilogy(imex_proc[k][i], imex_time[k][i], marker=imex_its[k][i], color=col[k], markersize=mksz)

    for i in range(len(newton_its[k])):
        plt.semilogy(newton_proc[k][i], newton_time[k][i], marker=newton_its[k][i], color=col[k], markersize=mksz)

plt.legend(legend)
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('wallclock time [s]', fontsize=12)
plt.grid('gray')
plt.xticks([0, 1, 2, 3, 4, 5, 6], [1, 2, 4, 8, 16, 32, 64])
plt.tight_layout()
plt.show()
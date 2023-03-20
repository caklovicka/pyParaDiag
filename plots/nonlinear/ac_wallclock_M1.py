import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mksz = 16
lw = 2
col = sns.color_palette("hls", 3)

file = ['data/ac1_0.dat']
legend = ['1e-6 (imex)', '1e-6 (newton)']

imex_proc = []
imex_time = []
imex_its = []
newton_proc = []
newton_time = []
newton_its = []
plt.figure(figsize=(5, 4), dpi=150)


for k in range(len(file)):
    # beta | nproc | time | tot iters | convergence
    table = np.loadtxt(file[k], delimiter='|', skiprows=3, usecols=[1, 2, 5, 8, 12])

    imex_proc.append([])
    imex_time.append([])
    imex_its.append([])
    newton_proc.append([])
    newton_time.append([])
    newton_its.append([])

    for i in range(table.shape[0]):

        if table[i, 4] == 0:
            continue

        rolling = 128 / table[i, 1]

        if table[i, 0] == 0:
            imex_proc[k].append(np.log2(table[i, 1]))
            imex_time[k].append(table[i, 2])
            imex_its[k].append('$' + str(round(table[i, 3] / rolling)) + '$')

        elif table[i, 0] == 1:
            newton_proc[k].append(np.log2(table[i, 1]))
            newton_time[k].append(table[i, 2])
            newton_its[k].append('$' + str(round(table[i, 3] / rolling)) + '$')

    plt.semilogy(imex_proc[k], imex_time[k], ':', color=col[k])
    plt.semilogy(newton_proc[k], newton_time[k], '-', color=col[k])

for k in range(len(file)):
    for i in range(len(imex_its[k])):
        plt.semilogy(imex_proc[k][i], imex_time[k][i], marker=imex_its[k][i], color=col[k], markersize=mksz, linewidth=2)

    for i in range(len(newton_its[k])):
        plt.semilogy(newton_proc[k][i], newton_time[k][i], marker=newton_its[k][i], color=col[k], markersize=mksz, linewidth=2)


plt.legend(legend)
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('time[s]', fontsize=12)
#plt.ylim([10, 10**4])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 4, 8, 16, 32, 64, 128])
plt.grid('gray')
plt.tight_layout()
plt.show()

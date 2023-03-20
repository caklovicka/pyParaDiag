
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

K = 2
mksz = 16
lw = 2
col = sns.color_palette("hls", 3)

file = ['data/ac2_0.dat', 'data/ac2_2.dat']
legend = ['1e-10 (imex)', '1e-10 (newton)', '1e-10 (imex) + coll', '1e-10 (newton) + coll']
col[0] = 'gray'

imex_proc = []
imex_time = []
imex_its = []
newton_proc = []
newton_time = []
newton_its = []
plt.figure(figsize=(5, 4), dpi=150)

for k in range(K):
    # beta | nproc | rolling | time | tot iters | convergence
    table = np.loadtxt(file[k], delimiter='|', skiprows=3, usecols=[1, 2, 3, 5, 8, 12])

    imex_proc.append([])
    imex_time.append([])
    imex_its.append([])
    newton_proc.append([])
    newton_time.append([])
    newton_its.append([])

    for i in range(table.shape[0]):

        if table[i, 5] == 0:
            continue

        rolling = table[i, 2]

        if table[i, 0] == 0:
            imex_proc[k].append(np.log2(table[i, 1]))
            imex_time[k].append(table[i, 3])
            imex_its[k].append('$' + str(round(table[i, 4] / rolling)) + '$')

        elif table[i, 0] == 1:
            newton_proc[k].append(np.log2(table[i, 1]))
            newton_time[k].append(table[i, 3])
            newton_its[k].append('$' + str(round(table[i, 4] / rolling)) + '$')

    plt.semilogy(imex_proc[k], imex_time[k], ':', color=col[k])
    plt.semilogy(newton_proc[k], newton_time[k], '-', color=col[k])

for k in range(K):
    for i in range(len(imex_its[k])):
        plt.semilogy(imex_proc[k][i], imex_time[k][i], marker=imex_its[k][i], color=col[k], markersize=mksz, linewidth=2)

    for i in range(len(newton_its[k])):
        plt.semilogy(newton_proc[k][i], newton_time[k][i], marker=newton_its[k][i], color=col[k], markersize=mksz, linewidth=2)


plt.legend(legend)
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('time[s]', fontsize=12)

xx = []
yy = []
for k in range(K):
    xx += imex_proc[k]
    xx += newton_proc[k]

for x in xx:
    yy.append(int(2 ** x))
plt.xticks(xx, yy)

plt.grid('gray')
plt.tight_layout()
plt.show()

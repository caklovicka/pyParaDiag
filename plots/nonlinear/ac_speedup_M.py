
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mksz = 10
lw = 2
col = sns.color_palette("hls", 3)

M = 3
if M == 1:
    file = ['data/ac1_0.dat']
    #file = ['data/ac_small_stol.dat']
    #file = ['data/ac_mixed_precision_M1.dat']
    legend = ['1e-6 (IMEX)', '1e-6 (Newton)']
    K = 1

elif M == 2:
    file = ['data/ac2_0.dat', 'data/ac2_2.dat']
    #file = ['data/ac2_0.dat', 'data/ac_mixed_precision_M2.dat']
    legend = ['1e-10 (IMEX)', '1e-10 (Newton)', '1e-10 (IMEX) + coll', '1e-10 (Newton) + coll']
    col[0] = 'gray'

elif M == 3:
    file = ['data/ac3_0.dat', 'data/ac3_2.dat']
    legend = ['1e-13 (IMEX)', '1e-13 (Newton)', '1e-13 (IMEX) + coll', '1e-13 (Newton) + coll']
    col = sns.color_palette("hls", 3)[-2:]
    col[0] = 'gray'

K = len(file)
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

    if 0 in imex_proc[0]:
        print(imex_proc[0])
        imex_seq = imex_time[0][imex_proc[0].index(0)]

    if 0 in newton_proc[0]:
        newton_seq = newton_time[0][newton_proc[0].index(0)]

    if M == 3:
        imex_seq = newton_seq

    if k <= K:
        plt.semilogy(imex_proc[k], imex_seq/np.array(imex_time[k]), '^:', color=col[k], linewidth=lw, markersize=mksz)
        plt.semilogy(newton_proc[k], newton_seq/np.array(newton_time[k]), 'v-', color=col[k], linewidth=lw, markersize=mksz)

plt.legend(legend)
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('speedup', fontsize=12)

xx = []
yy = []
for k in range(K):
    xx += imex_proc[k]
    xx += newton_proc[k]

for x in xx:
    yy.append(int(2 ** x))
plt.xticks(xx, yy)
plt.ylim([0.8, 60])
plt.xlim([-0.5, 8])

plt.grid('gray')
plt.tight_layout()
plt.show()

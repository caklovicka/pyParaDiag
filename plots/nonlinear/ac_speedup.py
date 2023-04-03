import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# TEST = 1 or 2
TEST = 1

mksz = 10
lw = 2
col = sns.color_palette("hls", 3)

#file = ['data/ac001_1.dat', 'data/ac001_2.dat', 'data/ac001_3.dat']
#file = ['data/ac_small_dt.dat']
file = ['data/ac001_1_optimal_alpha.dat']
K = len(file)
legend = ['1e-5 (imex)', '1e-5 (newton)', '1e-9 (imex)', '1e-9 (newton)', '1e-12 (imex)', '1e-12 (newton)']

imex_proc = []
imex_time = []
imex_its = []
newton_proc = []
newton_time = []
newton_its = []

imex_seq_time = []
newton_seq_time = []

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

        if table[i, 0] == 0:
            imex_proc[k].append(table[i, 1])
            imex_time[k].append(table[i, 2])
            imex_its[k].append('$' + str(round(table[i, 3] / rolling)) + '$')
            if table[i, 1] == 1:
                imex_seq_time.append(table[i, 2])

        elif table[i, 0] == 1:
            newton_proc[k].append(table[i, 1])
            newton_time[k].append(table[i, 2])
            newton_its[k].append('$' + str(round(table[i, 3] / rolling)) + '$')
            if table[i, 1] == 1:
                newton_seq_time.append(table[i, 2])

for k in range(K):
    plt.semilogy(np.log2(np.array(imex_proc[k])), imex_seq_time[k]/np.array(imex_time[k]), 'v:', color=col[k], linewidth=lw, markersize=mksz)
    if len(newton_proc[k]) > 0:
        plt.semilogy(np.log2(np.array(newton_proc[k])), newton_seq_time[k]/np.array(newton_time[k]), '^-', color=col[k], linewidth=lw, markersize=mksz)

plt.legend(legend)
plt.xlabel('total number of cores', fontsize=12)
plt.ylabel('speedup', fontsize=12)
plt.grid('gray')
plt.xticks([0, 1, 2, 3, 4, 5, 6], [1, 2, 4, 8, 16, 32, 64])

plt.legend(legend)
plt.tight_layout()
plt.show()

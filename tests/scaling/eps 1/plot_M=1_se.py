import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

K = 1
mksz = 15
col = sns.color_palette("bright", 2 * K)

legend = []
imex_proc = []
imex_time = []
imex_its = []
newton_proc = []
newton_time = []
newton_its = []

imex_seq_time = []
newton_seq_time = []

for k in range(K):
    # beta | nproc | time | tot iters | convergence
    print(k)
    table = np.loadtxt('output{}/000000/result/result.dat'.format(k + 1), delimiter='|', skiprows=3, usecols=[1, 2, 5, 8, 12])

    legend.append('imex ' + str(k + 1))
    legend.append('newton ' + str(k + 1))

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

plt.figure(figsize=(10, 5))
plt.subplot(121)
for k in range(K):
    print(imex_proc)
    plt.semilogx(imex_proc[k], np.log2(imex_seq_time[k]/np.array(imex_time[k])), 'X:', color=col[k])
    plt.semilogx(newton_proc[k], np.log2(newton_seq_time[k]/np.array(newton_time[k])), 'X-', color=col[k])

plt.legend(legend)
plt.xlabel('total number of cores')
plt.ylabel('log2(speedup)')
plt.grid('gray')
plt.xticks([1, 2, 4, 8, 16, 32, 64, 128], [1, 2, 4, 8, 16, 32, 64, 128])

plt.subplot(122)
for k in range(K):
    print(imex_proc)
    plt.semilogx(imex_proc[k], imex_seq_time[k] / (np.array(imex_time[k] * np.array(imex_proc[k]))), 'X:', color=col[k])
    plt.semilogx(newton_proc[k], newton_seq_time[k] / (np.array(newton_time[k]) * np.array(newton_proc[k])), 'X-', color=col[k])

plt.legend(legend)
plt.xlabel('total number of cores')
plt.ylabel('efficiency')
plt.grid('gray')
plt.xticks([1, 2, 4, 8, 16, 32, 64, 128], [1, 2, 4, 8, 16, 32, 64, 128])
plt.tight_layout()
plt.show()

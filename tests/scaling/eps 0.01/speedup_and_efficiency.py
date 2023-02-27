import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

K = 3
mksz = 20
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
    # beta | nproc | time | tot iters | convergence | diff
    table = np.loadtxt('output{}/000000/result/result.dat'.format(k + 1), delimiter='|', skiprows=3, usecols=[1, 2, 5, 8, 12, 13])

    legend.append('imex ' + str(k + 1))
    legend.append('newton ' + str(k + 1))

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

plt.figure(figsize=(10, 5))
plt.subplot(121)
for k in range(K):
    print(imex_proc)
    plt.semilogx(imex_proc[k], imex_seq_time[k]/np.array(imex_time[k]), 'X:', color=col[k])
    plt.semilogx(newton_proc[k], newton_seq_time[k]/np.array(newton_time[k]), 'X-', color=col[k])

plt.legend(legend)
plt.xlabel('cores')
plt.ylabel('speedup')
plt.grid('gray')
plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])

plt.subplot(122)
for k in range(K):
    print(imex_proc)
    plt.semilogx(imex_proc[k], imex_seq_time[k] / (np.array(imex_time[k] * np.array(imex_proc[k]))), 'X:', color=col[k])
    plt.semilogx(newton_proc[k], newton_seq_time[k] / (np.array(newton_time[k]) * np.array(newton_proc[k])), 'X-', color=col[k])

plt.legend(legend)
plt.xlabel('cores')
plt.ylabel('efficiency')
plt.grid('gray')
plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])

plt.show()

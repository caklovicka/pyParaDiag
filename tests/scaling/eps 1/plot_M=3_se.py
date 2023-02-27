
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

K = 2
mksz = 15
col = ['gray'] + sns.color_palette("bright", 2 * K)[2:]

legend = []
imex_proc = []
imex_time = []
imex_its = []
newton_proc = []
newton_time = []
newton_its = []

legend.append('imex 3')
legend.append('newton 3')
legend.append('imex 3 + coll')
legend.append('newton 3 + coll')

# missing sequential time for imex!
imex_seq_time = []
newton_seq_time = []

run = [0, 2]

for k in range(K):
    # beta | nproc | rolling | time | tot iters | convergence
    table = np.loadtxt('output3/00000{}/result/result.dat'.format(run[k]), delimiter='|', skiprows=4, usecols=[1, 2, 3, 5, 8, 12])

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
            imex_proc[k].append(table[i, 1])
            imex_time[k].append(table[i, 3])
            imex_its[k].append('$' + str(round(table[i, 4] / rolling)) + '$')

        elif table[i, 0] == 1:
            newton_proc[k].append(table[i, 1])
            newton_time[k].append(table[i, 3])
            newton_its[k].append('$' + str(round(table[i, 4] / rolling)) + '$')


plt.figure(figsize=(10, 5))
plt.subplot(121)
for k in range(K):
    print(imex_proc)
    print(imex_seq_time)
    print(imex_time)
    plt.semilogx(imex_proc[k], imex_seq_time[k]/np.array(imex_time[k]), 'X:', color=col[k])
    plt.semilogx(newton_proc[k], newton_seq_time[k]/np.array(newton_time[k]), 'X-', color=col[k])

plt.legend(legend)
plt.xlabel('total number of cores')
plt.ylabel('speedup')
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
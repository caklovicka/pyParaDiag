import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

K = 2
mksz=20
col = sns.color_palette("bright", 2 * K)

legend = []
imex_proc = []
imex_time = []
imex_its = []
newton_proc = []
newton_time = []
newton_its = []

for k in range(K):
    # beta | nproc | time | tot iters | convergence
    table = np.loadtxt('output{}/000000/result/result.dat'.format(k + 1), delimiter='|', skiprows=3, usecols=[1, 2, 5, 8, 12])

    legend.append('imex_' + str(k + 1))
    legend.append('newton_' + str(k + 1))

    imex_proc.append([])
    imex_time.append([])
    imex_its.append([])
    newton_proc.append([])
    newton_time.append([])
    newton_its.append([])

    for i in range(table.shape[0]):
        if table[i, 4] == 0:
            continue
        if table[i, 0] == 0:
            imex_proc[k].append(table[i, 1])
            imex_time[k].append(table[i, 2])
            imex_its[k].append('$' + str(int(table[i, 3])) + '$')

        elif table[i, 0] == 1:
            newton_proc[k].append(table[i, 1])
            newton_time[k].append(table[i, 2])
            newton_its[k].append('$' + str(int(table[i, 3])) + '$')

    plt.semilogx(imex_proc[k], imex_time[k], ':', color=col[k])
    plt.semilogx(newton_proc[k], newton_time[k], '--', color=col[k])

for k in range(K):
    for i in range(len(imex_its[k])):
        plt.plot(imex_proc[k][i], imex_time[k][i], marker=imex_its[k][i], color=col[k], markersize=mksz)

    for i in range(len(newton_its[k])):
        plt.plot(newton_proc[k][i], newton_time[k][i], marker=newton_its[k][i], color=col[k], markersize=mksz)


plt.legend(legend)
plt.xlabel('cores')
plt.ylabel('time[s]')
plt.title('eps = 0.01')
plt.xticks([1, 2, 4, 8, 16, 64], [1, 2, 4, 8, 16, 64])
plt.show()

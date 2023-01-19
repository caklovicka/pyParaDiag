
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

K = 2
mksz = 15
col = sns.color_palette("bright", 2 * K)

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
        print(table[i, 1])

        if table[i, 0] == 0:
            imex_proc[k].append(np.log2(table[i, 1]))
            imex_time[k].append(table[i, 3])
            imex_its[k].append('$' + str(round(table[i, 4] / rolling)) + '$')

        elif table[i, 0] == 1:
            newton_proc[k].append(np.log2(table[i, 1]))
            newton_time[k].append(table[i, 3])
            newton_its[k].append('$' + str(round(table[i, 4] / rolling)) + '$')

    plt.semilogy(imex_proc[k], imex_time[k], ':', color=col[k])
    plt.semilogy(newton_proc[k], newton_time[k], '--', color=col[k])

for k in range(K):
    for i in range(len(imex_its[k])):
        plt.semilogy(imex_proc[k][i], imex_time[k][i], marker=imex_its[k][i], color=col[k], markersize=mksz)

    for i in range(len(newton_its[k])):
        plt.semilogy(newton_proc[k][i], newton_time[k][i], marker=newton_its[k][i], color=col[k], markersize=mksz)


plt.legend(legend)
plt.xlabel('cores')
plt.ylabel('time[s]')
#plt.title('eps = 1')

xx = []
yy = []
for k in range(K):
    xx += imex_proc[k]
    xx += newton_proc[k]

for x in xx:
    yy.append(int(2 ** x))
plt.xticks(xx, yy)

plt.show()

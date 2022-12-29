import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# output1
#########
# run0: petsc scaling (1, 2, 4, 8, 16, 32, 64)
# run2: petsc(32), 64/n, k = 1e-2
# run3: petsc(64), 64/n, k = 1e-2
# run4: petsc(32), 32/n, k = 1e-2
# run5: petsc(16), 16/n, k = 1e-2

# output2
#########
# run0: petsc(64), 64/n, k = 1e-1


legend = []
petsc_proc = []
petsc_time = []
custom_lines = []

# just PETSc
# nproc | time
table = np.loadtxt('output1/000000/result/result.dat', delimiter='|', skiprows=3, usecols=[1, 2])
seq_time = table[0, 1]

for i in range(table.shape[0]):
    petsc_proc.append(np.log2(table[i, 0]))
    petsc_time.append(table[i, 1])

plt.semilogy(petsc_proc, petsc_time, 'X-', color='gray', markersize=10)
plt.semilogy(petsc_proc, seq_time / (2 ** np.array(petsc_proc)), 'X:', color='gray', markersize=10)

# PinT + PETSc
runs = [3, 4, 5]
K = len(runs)
runs = [3, 4, 5]
mksz = 40
col = sns.color_palette("bright", K)

pint_petsc_proc = []
pint_petsc_time = []
pint_petsc_iters = []

custom_lines.append(Line2D([0], [0], color='gray', linestyle='-'))
custom_lines.append(Line2D([0], [0], color='gray', linestyle=':'))
legend.append('just petsc')
legend.append('petsc ideal')

for k in range(K):
    # nproc | proc_col | time | tot iters
    table = np.loadtxt('output1/00000{}/result/result.dat'.format(runs[k]), delimiter='|', skiprows=3, usecols=[1, 2, 5, 8])

    pint_petsc_proc.append([])
    pint_petsc_time.append([])
    pint_petsc_iters.append([])

    proc_col = table[0, 1]
    legend.append('petsc({}) + pint '.format(int(proc_col)))
    custom_lines.append(Line2D([0], [0], color=col[k], linestyle='--'))

    pint_petsc_proc[k].append(np.log2(int(proc_col)))
    petsc_time_seq = petsc_time[int(pint_petsc_proc[k][-1])]
    pint_petsc_time[k].append(petsc_time_seq)
    #pint_petsc_iters[k].append(32)
    pint_petsc_iters[k].append(1)

    for i in range(table.shape[0]):
        pint_petsc_proc[k].append(np.log2(table[i, 0]))
        pint_petsc_time[k].append(table[i, 2])

        # pint_petsc_iters[k].append('$' + str(int(table[i, 3])) + '$')
        rolling = 32 / (table[i, 0] / table[i, 1])
        it = table[i, 3] / rolling
        if float(int(it)) == it:
            it = int(it)
        pint_petsc_iters[k].append(it)

    for i in range(len(pint_petsc_iters[k])):
        if int(pint_petsc_iters[k][i]) == pint_petsc_iters[k][i]:
            msize = mksz // 2
        else:
            msize = mksz
        marker = '$' + str(pint_petsc_iters[k][i]) + '$'
        plt.semilogy(pint_petsc_proc[k][i], pint_petsc_time[k][i], color=col[k], markersize=msize, marker=marker)

    plt.semilogy(pint_petsc_proc[k], pint_petsc_time[k], linestyle='--', color=col[k])
    plt.semilogy(pint_petsc_proc[k], seq_time / (2 ** np.array(pint_petsc_proc[k])), 'X:', color=col[k], markersize=10)


plt.legend(custom_lines, legend)
print(custom_lines)
print(legend)

plt.xlabel('cores')
plt.ylabel('time[s]')
plt.title('scaling PETSc on Boltzmann (32 times steps, M=1)')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
plt.show()

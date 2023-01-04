import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

legend = []
petsc_proc = []
petsc_time = []
custom_lines = []

# just PETSc
# nproc | time
table = np.loadtxt('output1/000000/result/result.dat', delimiter='|', skiprows=3, usecols=[1, 2])
seq_time = table[0, 1]


# PinT + PETSc
runs = [[3, 4, 5], [1], [0]]
output = [1, 2, 3]
knudsen = [1e-2, 1e-1, 1e-4]

mksz = 40
col = sns.color_palette("bright", 3)
linestyle = [':', '--', '-']
lw = 2

pint_petsc_proc = []
pint_petsc_time = []
pint_petsc_iters = []

for kk in range(len(output)):
    pint_petsc_proc.append([])
    pint_petsc_time.append([])
    pint_petsc_iters.append([])

    for k in range(len(runs[kk])):
        # nproc | proc_col | time | tot iters
        table = np.loadtxt('output{}/00000{}/result/result.dat'.format(output[kk], runs[kk][k]), delimiter='|', skiprows=3, usecols=[1, 2, 5, 8])

        pint_petsc_proc[kk].append([])
        pint_petsc_time[kk].append([])
        pint_petsc_iters[kk].append([])

        proc_col = table[0, 1]
        legend.append('petsc({}) + pint, Kn = {}'.format(int(proc_col), knudsen[kk]))
        custom_lines.append(Line2D([0], [0], color=col[k], linestyle=linestyle[kk]))

        for i in range(table.shape[0]):
            pint_petsc_proc[kk][k].append(np.log2(table[i, 0]))
            pint_petsc_time[kk][k].append(table[i, 2])

            # pint_petsc_iters[k].append('$' + str(int(table[i, 3])) + '$')
            rolling = 32 / (table[i, 0] / table[i, 1])
            it = table[i, 3] / rolling
            if float(int(it)) == it:
                it = int(it)
            pint_petsc_iters[kk][k].append(it)

        for i in range(len(pint_petsc_iters[kk][k])):
            if int(pint_petsc_iters[kk][k][i]) == pint_petsc_iters[kk][k][i]:
                msize = mksz // 2
            else:
                msize = mksz
            marker = '$' + str(pint_petsc_iters[kk][k][i]) + '$'
            plt.semilogy(pint_petsc_proc[kk][k][i], pint_petsc_time[kk][k][i], color=col[k], markersize=msize, marker=marker)

        plt.semilogy(pint_petsc_proc[kk][k], pint_petsc_time[kk][k], linestyle='--', color=col[k], linewidth=lw)
        plt.semilogy(pint_petsc_proc[kk][k], seq_time / (2 ** np.array(pint_petsc_proc[kk][k])), 'X:', color=col[k], markersize=10, linewidth=lw)


plt.legend(custom_lines, legend)
print(custom_lines)
print(legend)

plt.xlabel('cores')
plt.ylabel('time[s]')
plt.title('scaling PETSc on Boltzmann (32 times steps, M=1)')
plt.xticks([5, 6, 7, 8, 9, 10, 11], [32, 64, 128, 256, 512, 1024, 2048])
plt.show()

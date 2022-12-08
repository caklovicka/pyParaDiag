import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

K = 1
mksz = 20
col = sns.color_palette("bright", 2 * K)

legend = []
petsc_proc = []
petsc_time = []

for k in range(K):
    # nproc | time
    table = np.loadtxt('output{}/000000/result/result.dat'.format(k + 1), delimiter='|', skiprows=3, usecols=[1, 2])
    print(table)

    legend.append('petsc ' + str(k + 1))
    legend.append('petsc ' + str(k + 1) + ' ideal')

    petsc_proc.append([])
    petsc_time.append([])

    for i in range(table.shape[0]):
        petsc_proc[k].append(np.log2(table[i, 0]))
        petsc_time[k].append(table[i, 1])

    plt.semilogy(petsc_proc[k], petsc_time[k], 'X-', color='gray', markersize=10)
    plt.semilogy(petsc_proc[k], petsc_time[k][0] / (2 ** np.array(petsc_proc[k])), 'X:', color='gray', markersize=10)

plt.legend(legend)
plt.xlabel('cores')
plt.ylabel('time[s]')
plt.title('scaling PETSc on Boltzmann (32 times steps, M=1)')
plt.xticks([0, 1, 2, 3, 4, 5, 6], [1, 2, 4, 8, 16, 32, 64])
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

NAME = 'heat'
path_heat = ['heat1_strong/output/000001/result/result.dat', 'heat2_strong/output/000002/result/result.dat', 'heat3_strong/output/000033/result/result.dat']
path_adv = ['adv1_strong/output/000009/result/result.dat', 'adv2_strong/output/000004/result/result.dat', 'adv3_strong/output/000005/result/result.dat']

if NAME == 'adv':
    path = path_adv.copy()
if NAME == 'heat':
    path = path_heat.copy()

# nproc | tot_time | iters
eq = []
for p in path:
    eq.append(np.loadtxt(p, delimiter='|', usecols=[0, 3, 7], skiprows=3))

seqT = []
# sort runs
for i in range(len(eq)):
    indices = np.argsort(eq[i][:, 0])
    eq[i] = eq[i][indices, :]
    seqT.append(eq[i][0, 1])

ticks = []
labels = []
n = eq[0].shape[0]
marks = 16
lw = 2
linst = ['dotted', 'dashed', 'dashdot']
custom_lines = []
col = sns.color_palette("bright", len(eq))

for i in range(len(eq)):
    eq[i][:, 1] = seqT[i] / eq[i][:, 1]

    plt.plot(np.log2(eq[i][:, 0]), eq[i][:, 1], linestyle=linst[i], linewidth=lw, color=col[i])
    custom_lines.append(Line2D([0], [0], linestyle=linst[i], linewidth=lw, color=col[i]))
    ticks += list(np.log2(eq[i][:, 0]))
    labels += list(eq[i][:, 0])

for i in range(len(eq)):
    for nn in range(n):
        m = int(eq[i][nn, 2])
        plt.plot(np.log2(eq[i][nn, 0]), eq[i][nn, 1], marker="$" + str(m) + "$", markersize=marks, color=col[i])

labels = [int(l) for l in labels]
plt.xticks(ticks, labels)
names = ['1e-5, M=1', '1e-9, M=2', '1e-12, M=3', 'k iterations']
custom_lines.append(Line2D([0], [0], marker="$k$", markersize=10, color='gray'))
plt.legend(custom_lines, names, loc='upper left')
plt.grid(True, color='gainsboro')
plt.ylabel('speedup')
plt.xlabel('number of cores')
plt.ylim([0, 15])
# plt.show()
plt.savefig('strong_plots/Stepparallelstrong' + NAME, dpi=300, bbox_inches='tight')
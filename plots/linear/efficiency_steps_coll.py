import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

NAME = 'adv'
path_heat_seq = ['data/heat1_2.dat', 'data/heat2_4.dat', 'data/heat3_35.dat']
path_adv_seq = ['data/adv1_12.dat', 'data/adv2_7.dat', 'data/adv3_8.dat']

path_heat = ['data/heat1_2.dat', 'data/heat2_5.dat', 'data/heat3_36.dat']
path_adv = ['data/adv1_12.dat', 'data/adv2_8.dat', 'data/adv3_9.dat']

if NAME == 'adv':
    path = path_adv.copy()
    path_seq = path_adv_seq.copy()
if NAME == 'heat':
    path = path_heat.copy()
    path_seq = path_heat_seq.copy()

# nproc | tot_time | iters
eq = []
eq_seq = []
for p in path:
    eq.append(np.loadtxt(p, delimiter='|', usecols=[0, 3, 7], skiprows=3))
for p in path_seq:
    eq_seq.append(np.loadtxt(p, delimiter='|', usecols=[0, 3, 7], skiprows=3))

seqT = []
# sort runs
for i in range(len(eq)):
    indices = np.argsort(eq[i][:, 0])
    eq[i] = eq[i][indices, :]

    indices = np.argsort(eq_seq[i][:, 0])
    eq_seq[i] = eq_seq[i][indices, :]
    seqT.append(eq_seq[i][0, 1])

ticks = []
labels = []
n = eq[0].shape[0]
marks = 16
lw = 2
linst = ['dotted', 'dashed', 'dashdot']
custom_lines = []
col = sns.color_palette("bright", len(eq))

# ild plots in gray
for i in range(len(eq)):
    eq_seq[i][:, 1] = seqT[i] / eq_seq[i][:, 1]

    plt.plot(np.log2(eq_seq[i][:, 0]), eq_seq[i][:, 1]/eq_seq[i][:, 0], linestyle=linst[i], linewidth=lw, color='silver')
    ticks += list(np.log2(eq_seq[i][:, 0]))
    labels += list(eq_seq[i][:, 0])

for i in range(len(eq)):
    for nn in range(n):
        m = int(eq_seq[i][nn, 2])
        plt.plot(np.log2(eq_seq[i][nn, 0]), eq_seq[i][nn, 1]/eq_seq[i][nn, 0], marker="$" + str(m) + "$", markersize=marks, color='silver')

#new plots
for i in range(len(eq)):
    eq[i][:, 1] = seqT[i] / eq[i][:, 1]

    plt.plot(np.log2(eq[i][:, 0]), eq[i][:, 1]/eq[i][:, 0], linestyle=linst[i], linewidth=lw, color=col[i])
    custom_lines.append(Line2D([0], [0], linestyle=linst[i], linewidth=lw, color=col[i]))
    ticks += list(np.log2(eq[i][:, 0]))
    labels += list(eq[i][:, 0])

for i in range(len(eq)):
    for nn in range(n):
        m = int(eq[i][nn, 2])
        plt.plot(np.log2(eq[i][nn, 0]), eq[i][nn, 1]/eq[i][nn, 0], marker="$" + str(m) + "$", markersize=marks, color=col[i])

labels = [int(l) for l in labels]
plt.xticks(ticks, labels)
names = ['1e-5, M=1', '1e-9, M=2', '1e-12, M=3', 'k iterations']
custom_lines.append(Line2D([0], [0], marker="$k$", markersize=10, color='gray'))
plt.legend(custom_lines, names, loc='upper right')
plt.grid(True, color='gainsboro')
plt.ylabel('efficiency')
plt.xlabel('total number of cores')
plt.ylim([-0.1, 1.1])
plt.tight_layout()
plt.show()
#plt.savefig('strong_plots/Stepcollparallelstrong' + NAME, dpi=300, bbox_inches='tight')
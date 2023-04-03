import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

NAME = 'steady_state'
path_heat_seq = ['data/heat1_2.dat', 'data/heat2_4.dat', 'data/heat3_35.dat']
path_adv_seq = ['data/adv1_12.dat', 'data/adv2_7.dat', 'data/adv3_8.dat']

path_heat = ['data/heat1_2.dat', 'data/heat2_5.dat', 'data/heat3_36.dat']
path_adv = ['data/adv1_12.dat', 'data/adv2_8.dat', 'data/adv3_9.dat']

adv_small_dt_seq = ['data/adv_small_dt_steps.dat']
adv_small_dt = ['data/adv_small_dt_steps_coll.dat']

heat_steady_state_seq = ['data/heat_steady_state.dat']
heat_steady_state = ['data/heat_steady_state.dat']

plt.figure(figsize=(5, 4), dpi=150)
linst = ['dotted', 'dashed', 'dashdot']
names = ['1e-5, M=1', '1e-9, M=2', '1e-12, M=3', 'k iterations']
col = sns.color_palette("hls", 3)
markers = []
ticks = []
labels = []
marks = 16
lw = 2
custom_lines = []

if NAME == 'adv':
    path = path_adv.copy()
    path_seq = path_adv_seq.copy()
if NAME == 'heat':
    path = path_heat.copy()
    path_seq = path_heat_seq.copy()
if NAME == 'small_dt':
    path = adv_small_dt.copy()
    path_seq = adv_small_dt_seq.copy()
    linst = ['dashdot']
    names = ['1e-12, M=3', 'k iterations']
    col = col[-1:]
    markers = ['>']
    marks = 10
if NAME == 'steady_state':
    path = heat_steady_state.copy()
    path_seq = heat_steady_state_seq.copy()
    names = ['1e-6, M=1', 'k iterations']
    #markers = ['>']
    #marks = 10

# nproc | tot_time | iters
eq = []
eq_seq = []
for p in path:
    eq.append(np.loadtxt(p, delimiter='|', usecols=[0, 3, 7], skiprows=3))
for p in path_seq:
    eq_seq.append(np.loadtxt(p, delimiter='|', usecols=[0, 3, 7], skiprows=3))

n = eq[0].shape[0]
seqT = []
# sort runs
for i in range(len(eq)):
    indices = np.argsort(eq[i][:, 0])
    eq[i] = eq[i][indices, :]

    indices = np.argsort(eq_seq[i][:, 0])
    eq_seq[i] = eq_seq[i][indices, :]
    seqT.append(eq_seq[i][0, 1])

# old plots in gray
for i in range(len(eq)):
    eq_seq[i][:, 1] = seqT[i] / eq_seq[i][:, 1]

    if markers != ['>']:
        plt.semilogy(np.log2(eq_seq[i][:, 0]), eq_seq[i][:, 1], linestyle=linst[i], linewidth=lw, color='silver')
    else:
        print('tu sam')
        plt.semilogy(np.log2(eq_seq[i][:, 0]), eq_seq[i][:, 1], linestyle=linst[i], linewidth=lw, color='silver', marker=markers[0], markersize=marks)
        print('tu sam')

    ticks += list(np.log2(eq_seq[i][:, 0]))
    labels += list(eq_seq[i][:, 0])

if markers != ['>']:
    for i in range(len(eq)):
        for nn in range(n):
            m = int(eq_seq[i][nn, 2])
            plt.semilogy(np.log2(eq_seq[i][nn, 0]), eq_seq[i][nn, 1], marker="$" + str(m) + "$", markersize=marks, color='silver')

#new plots
for i in range(len(eq)):
    eq[i][:, 1] = seqT[i] / eq[i][:, 1]

    if markers != ['>']:
        plt.semilogy(np.log2(eq[i][:, 0]), eq[i][:, 1], linestyle=linst[i], linewidth=lw, color=col[i])
    else:
        plt.semilogy(np.log2(eq[i][:, 0]), eq[i][:, 1], linestyle=linst[i], linewidth=lw, color=col[i], marker=markers[0], markersize=marks)

    custom_lines.append(Line2D([0], [0], linestyle=linst[i], linewidth=lw, color=col[i]))
    ticks += list(np.log2(eq[i][:, 0]))
    labels += list(eq[i][:, 0])

if markers != ['>']:
    for i in range(len(eq)):
        plt.semilogy(np.log2(eq_seq[i][nn, 0]), eq_seq[i][nn, 1], marker="$" + str(m) + "$", markersize=marks, color='silver')
        for nn in range(n):
            m = int(eq[i][nn, 2])
            plt.semilogy(np.log2(eq[i][nn, 0]), eq[i][nn, 1], marker="$" + str(m) + "$", markersize=marks, color=col[i])

labels = [int(l) for l in labels]
plt.xticks(ticks, labels)
custom_lines.append(Line2D([0], [0], marker="$k$", markersize=10, color='gray'))
plt.legend(custom_lines, names, loc='upper left')
plt.grid(True, color='gainsboro')
plt.ylabel('speedup', fontsize=12)
plt.xlabel('total number of cores', fontsize=12)
#plt.ylim([-7, 26])
plt.tight_layout()
plt.show()
#plt.savefig('strong_plots/Stepcollparallelstrong' + NAME, dpi=300, bbox_inches='tight')
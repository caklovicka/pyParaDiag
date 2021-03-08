import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

'plotting a fully serial and interval-parallel speedup'

NAME = 'Euler'

if NAME == 'Heat':
    path3 = ['heat1_strong/output/000000/result/result.dat', 'heat2_strong/output/000000/result/result.dat',
             'heat3_strong/output/000005/result/result.dat']
if NAME == 'Advection':
    path3 = ['adv1_strong/output/000001/result/result.dat', 'adv2_strong/output/000000/result/result.dat',
             'adv3_strong/output/000000/result/result.dat']
if NAME == 'Schrodinger':
    path3 = ['schro1_strong/output/000000/result/result.dat', 'schro2_strong/output/000000/result/result.dat',
             'schro3_strong/output/000000/result/result.dat']

if NAME == 'Euler':
    path3 = ['schro1_strong/output/000000/result/result.dat', 'heat1_strong/output/000000/result/result.dat',
             'adv1_strong/output/000001/result/result.dat']

# nproc | rolling | time_intervals | tot_time | max_paralpha_iters
eq3 = []
for i in range(len(path3)):
    eq3.append(np.loadtxt(path3[i], delimiter='|', usecols=[0, 1, 2, 3, 7], skiprows=3))

no_runs = len(eq3[0]) - 1
speedup = np.ones((no_runs, len(eq3)))
its = np.zeros((no_runs, len(eq3)))

for run in range(len(eq3)):
    for subrun in range(no_runs + 1):
        # if its a parallel run
        if eq3[run][subrun, 0] > 1:
            row = int(np.log2(eq3[run][subrun, 2])) - 2
            speedup[row, run] /= eq3[run][subrun, 3]
            its[row, run] = int(eq3[run][subrun, 4])
        else:
            speedup[:, run] *= eq3[run][subrun, 3]

marks = 16
lw = 2
col = sns.color_palette("bright", len(eq3))
linst = ['dotted', 'dashed', 'dashdot']
custom_lines = []
legend = []
proc = range(no_runs)
nproc = [4, 8, 16, 32, 64]
names = ['1e-5', '1e-9', '1e-12']

# speedup plot
for run in range(len(eq3)):
    plt.plot(np.log2(nproc), list(speedup[:, run]), linestyle=linst[run], linewidth=lw, color=col[run])
    custom_lines.append(Line2D([0], [0], linestyle=linst[run], linewidth=lw, color=col[run]))

# markers
for run in range(len(eq3)):
    for subrun in range(no_runs):
        m = int(its[subrun, run])
        plt.plot(np.log2(nproc[subrun]), speedup[subrun, run], marker="$" + str(m) + "$", markersize=marks, color=col[run])

custom_lines.append(Line2D([0], [0], marker="$k$", markersize=10, color='gray'))
names.append('k iterations')

if NAME != 'Euler':
    plt.legend(custom_lines, names, loc='upper left')
else:
    names[0] = 'Schrodinger'
    names[1] = 'Heat'
    names[2] = 'Advection'
    plt.legend(custom_lines, names, loc='upper left')
plt.xticks(np.log2(nproc), nproc)
plt.ylabel('speedup')
plt.xlabel('number of cores')
plt.ylim([0, 20])
plt.show()
# plt.savefig('strong_plots/Speedup_' + NAME + '_interval', dpi=300, bbox_inches='tight')


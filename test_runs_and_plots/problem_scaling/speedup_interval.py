import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

'plotting the a fully serial and interval-parallel speedup'

NAME = 'Advection'

if NAME == 'Heat':
    path3 = ['heat1/output/000000/result/result.dat', 'heat2/output/000000/result/result.dat',
             'heat3/output/000000/result/result.dat']
if NAME == 'Advection':
    path3 = ['adv1/output/000002/result/result.dat', 'adv2/output/000002/result/result.dat',
             'adv3/output/000002/result/result.dat']
if NAME == 'Schrodinger':
    path3 = ['schro1/output/000002/result/result.dat', 'schro2/output/000002/result/result.dat',
             'schro3/output/000002/result/result.dat']

# rolling | time_intervals | tot_time | paralpha_iters | tol
eq3 = []
for i in range(len(path3)):
    eq3.append(np.loadtxt(path3[i], delimiter='|', usecols=[0, 1, 2, 4, 9], skiprows=3))

no_runs = len(eq3[0]) // 2
speedup = np.ones((no_runs, len(eq3)))
its = np.zeros((no_runs, len(eq3)))

for run in range(len(eq3)):
    for subrun in range(2 * no_runs):
        # if rolling = 1
        if eq3[run][subrun, 0] == 1:
            row = int(np.log2(eq3[run][subrun, 1])) - 2
            speedup[row, run] /= eq3[run][subrun, 2]
            its[row, run] = int(eq3[run][subrun, 3])
        else:
            row = int(np.log2(eq3[run][subrun, 0])) - 2
            speedup[row, run] *= eq3[run][subrun, 2]

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

for run in range(len(eq3)):
    for subrun in range(no_runs):
        m = int(its[subrun, run])
        plt.plot(np.log2(nproc[subrun]), speedup[subrun, run], marker="$" + str(m) + "$", markersize=marks, color=col[run])

custom_lines.append(Line2D([0], [0], marker="$k$", markersize=10, color='gray'))
names.append('k iterations')

plt.legend(custom_lines, names, loc='upper left')
plt.xticks(np.log2(nproc), nproc)
plt.ylabel('speedup')
plt.xlabel('number of cores')
plt.ylim([0, 20])
# plt.title(NAME + ' equation')
# plt.show()
# fig = plt.gcf()
# fig.set_size_inches(4, 4)
plt.savefig('AAplots/speedup_' + NAME + '_interval', dpi=300, bbox_inches='tight')


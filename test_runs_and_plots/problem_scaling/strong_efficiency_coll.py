import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

'plotting the a coll-parallel and interval-coll-parallel speedup' \
'the gray plots are the fully serial and interval-parallel speedup'

NAME = 'Schrodinger'

if NAME == 'Heat':
    path2 = ['heat2_strong/output/000001/result/result.dat', 'heat3_strong/output/000032/result/result.dat']
    path3 = ['heat1_strong/output/000000/result/result.dat', 'heat2_strong/output/000000/result/result.dat',
             'heat3_strong/output/000005/result/result.dat']
if NAME == 'Advection':
    path2 = ['adv2_strong/output/000001/result/result.dat', 'adv3_strong/output/000001/result/result.dat']
    path3 = ['adv1_strong/output/000001/result/result.dat', 'adv2_strong/output/000000/result/result.dat',
             'adv3_strong/output/000000/result/result.dat']
if NAME == 'Schrodinger':
    path2 = ['schro2_strong/output/000001/result/result.dat', 'schro3_strong/output/000001/result/result.dat']
    path3 = ['schro1_strong/output/000000/result/result.dat', 'schro2_strong/output/000000/result/result.dat',
             'schro3_strong/output/000000/result/result.dat']

# nproc | rolling | time_intervals | tot_time | max_paralpha_iters
eq3 = []
for i in range(len(path3)):
    eq3.append(np.loadtxt(path3[i], delimiter='|', usecols=[0, 1, 2, 3, 7], skiprows=3))

# nproc | rolling | time_intervals | tot_time | max_paralpha_iters
eq2 = []
for i in range(len(path2)):
    eq2.append(np.loadtxt(path2[i], delimiter='|', usecols=[0, 1, 2, 3, 7], skiprows=3))

####################
# OLD PLOTS IN GRAY
####################

no_runs = len(eq3[0]) - 1
efficiency = np.ones((no_runs, len(eq3)))
its = np.zeros((no_runs, len(eq3)))

for run in range(len(eq3)):
    for subrun in range(no_runs + 1):
        # if its a parallel run
        if eq3[run][subrun, 0] > 1:
            row = int(np.log2(eq3[run][subrun, 2])) - 2
            efficiency[row, run] /= eq3[run][subrun, 3]
            efficiency[row, run] /= eq3[run][subrun, 0]
            its[row, run] = int(eq3[run][subrun, 4])
        else:
            efficiency[:, run] *= eq3[run][subrun, 3]

marks = 16
lw = 2
col = [sns.color_palette("bright", len(eq3))[0], 'silver', 'silver']
linst = ['dotted', 'dashed', 'dashdot']
proc = range(no_runs)
nproc = [4, 8, 16, 32, 64]

# speedup plot
for run in range(len(eq3) - 1, -1, -1):
    plt.plot(np.log2(nproc), list(efficiency[:, run]), linestyle=linst[run], linewidth=lw, color=col[run])

# markers
for run in range(len(eq3) - 1, -1, -1):
    for subrun in range(no_runs):
        m = int(its[subrun, run])
        plt.plot(np.log2(nproc[subrun]), efficiency[subrun, run], marker="$" + str(m) + "$", markersize=marks, color=col[run])

plt.xticks(np.log2(nproc), nproc)

####################
# NEW PLOTS
####################

no_runs = len(eq2[0]) - 1
efficiency = np.ones((no_runs, len(eq2)))
its = np.zeros((no_runs, len(eq2)))

for run in range(len(eq2)):
    for subrun in range(no_runs + 1):
        # if its a parallel run
        if eq2[run][subrun, 1] < 64:
            row = int(np.log2(eq2[run][subrun, 2])) - 2
            print(row)
            efficiency[row, run] /= eq2[run][subrun, 3]
            efficiency[row, run] /= eq2[run][subrun, 0]
            its[row, run] = int(eq2[run][subrun, 4])
        else:
            efficiency[:, run] *= eq2[run][subrun, 3]

marks = 16
lw = 2
col = sns.color_palette("bright", len(eq3))
linst = ['dashed', 'dashdot']
custom_lines = []
legend = []
proc = range(no_runs)
nproc = [[8, 16, 32, 64, 128], [12, 24, 16*3, 32*3, 64*3]]
nnproc = [4, 8, 16, 32, 64, 128, 12, 24, 16*3, 32*3, 64*3]
names = ['1e-5', '1e-9', '1e-12']
custom_lines.append(Line2D([0], [0], linestyle='dotted', linewidth=lw, color=col[0]))
col = col[1:]

# speedup plot
for run in range(len(eq2)):
    plt.plot(np.log2(nproc[run]), list(efficiency[:, run]), linestyle=linst[run], linewidth=lw, color=col[run])
    custom_lines.append(Line2D([0], [0], linestyle=linst[run], linewidth=lw, color=col[run]))

# markers
for run in range(len(eq2)):
    for subrun in range(no_runs):
        m = int(its[subrun, run])
        plt.plot(np.log2(nproc[run][subrun]), efficiency[subrun, run], marker="$" + str(m) + "$", markersize=marks, color=col[run])

custom_lines.append(Line2D([0], [0], marker="$k$", markersize=10, color='gray'))
names.append('k iterations')

plt.legend(custom_lines, names, loc='upper left')
plt.xticks(np.log2(nnproc), nnproc)
plt.ylabel('efficiency')
plt.xlabel('number of cores')
plt.ylim([0, 0.62])
# plt.show()
plt.savefig('strong_plots/Efficiency_' + NAME + '_coll', dpi=300, bbox_inches='tight')
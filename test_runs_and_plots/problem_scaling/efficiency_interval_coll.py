import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

'plotting the a fully serial and interval-coll-parallel efficiency' \
'the gray plots are the fully serial and interval-parallel efficiency'

NAME = 'Schrodinger'

if NAME == 'Heat':
    path2 = ['heat2/output/000001/result/result.dat', 'heat3/output/000001/result/result.dat']
    path3 = ['heat1/output/000000/result/result.dat', 'heat2/output/000000/result/result.dat',
             'heat3/output/000000/result/result.dat']
if NAME == 'Advection':
    path2 = ['adv2/output/000003/result/result.dat', 'adv3/output/000003/result/result.dat']
    path3 = ['adv1/output/000002/result/result.dat', 'adv2/output/000002/result/result.dat',
             'adv3/output/000002/result/result.dat']
if NAME == 'Schrodinger':
    path2 = ['schro2/output/000003/result/result.dat', 'schro3/output/000003/result/result.dat']
    path3 = ['schro1/output/000002/result/result.dat', 'schro2/output/000002/result/result.dat',
             'schro3/output/000002/result/result.dat']

# rolling | time_intervals | tot_time | paralpha_iters | tol
eq3 = []
for i in range(len(path3)):
    eq3.append(np.loadtxt(path3[i], delimiter='|', usecols=[0, 1, 2, 4, 9], skiprows=3))
# nproc | rolling | time_intervals | tot_time | paralpha_iters | tol
eq2 = []
for i in range(len(path2)):
    eq2.append(np.loadtxt(path2[i], delimiter='|', usecols=[0, 1, 2, 3, 5, 10], skiprows=3))

####################
# OLD PLOTS IN GRAY
####################

no_runs = len(eq3[0]) // 2
efficiency = np.ones((no_runs, len(eq3)))

for run in range(len(eq3)):
    for subrun in range(2 * no_runs):
        # if rolling = 1
        if eq3[run][subrun, 0] == 1:
            row = int(np.log2(eq3[run][subrun, 1])) - 2
            efficiency[row, run] /= eq3[run][subrun, 1]
            efficiency[row, run] /= eq3[run][subrun, 2]
        else:
            row = int(np.log2(eq3[run][subrun, 0])) - 2
            efficiency[row, run] *= eq3[run][subrun, 2]

legend = []
proc = range(no_runs)
labels = [4, 8, 16, 32, 64]
names = ['1e-5', '1e-9', '1e-12']
markers_in_use = set()
marks = 16
lw = 2
linst = ['dotted', 'dashed', 'dashdot']

# efficiency plot
for run in range(len(eq3)):
    plt.plot(np.log2(labels), list(efficiency[:, run]), linestyle=linst[run], linewidth=lw, color='silver')

for run in range(len(eq3)):
    for subrun in range(no_runs):
        m = int(eq3[run][subrun, 3])
        plt.plot(np.log2(labels[subrun]), efficiency[subrun, run], marker="$" + str(m) + "$", markersize=marks, color='silver')
        markers_in_use.add(int(eq3[run][subrun, 3]))

plt.xticks(np.log2(labels), labels)

####################
# new plot
####################

no_runs = len(eq3[0]) // 2
efficiency = np.ones((no_runs, len(path3)))
its = np.zeros((no_runs, len(eq3)))

for run in range(len(eq3)):
    for subrun in range(2 * no_runs):
        # if we are in seq run
        if eq3[run][subrun, 0] > 1:
            row = int(np.log2(eq3[run][subrun, 0])) - 2
            efficiency[row, run] *= eq3[run][subrun, 2]
        # if we are in a parallel run
        else:
            # if its euler
            if run == 0:
                row = int(np.log2(eq3[run][subrun, 1])) - 2
                efficiency[row, run] /= eq3[run][subrun, 2]
                efficiency[row, run] /= eq3[run][subrun, 1]
                its[row, run] = int(eq3[run][subrun, 3])

# its not euler
for run in range(len(eq3)):
    for subrun in range(2 * no_runs):
        if run == 0 or eq2[run - 1][subrun, 1] > 1:
            continue
        row = int(np.log2(eq2[run - 1][subrun, 2])) - 2
        efficiency[row, run] /= eq2[run - 1][subrun, 3]
        efficiency[row, run] /= eq2[run - 1][subrun, 0]
        its[row, run] = int(eq2[run - 1][subrun, 4])


##############
# PLOT
##############

marks = 16
lw = 2
col = sns.color_palette("bright", len(eq3))
linst = ['dotted', 'dashed', 'dashdot']
custom_lines = []
legend = []
proc = range(no_runs)
nproc = [[4, 8, 16, 32, 64], [8, 16, 32, 64, 128], [12, 24, 16*3, 32*3, 64*3]]
nnproc = [4, 8, 16, 32, 64, 8, 16, 32, 64, 128, 12, 24, 16*3, 32*3, 64*3]
names = ['1e-5', '1e-9', '1e-12']
markers_in_use = set()

# efficiency plot
for run in range(len(eq3)):
    plt.plot(np.log2(nproc[run]), list(efficiency[:, run]), linestyle=linst[run], linewidth=lw, color=col[run])
    custom_lines.append(Line2D([0], [0], linestyle=linst[run], linewidth=lw, color=col[run]))

for run in range(len(eq3)):
    for subrun in range(no_runs):
        m = int(its[subrun, run])
        plt.plot(np.log2(nproc[run][subrun]), efficiency[subrun, run], marker="$" + str(m) + "$", markersize=marks, color=col[run])

custom_lines.append(Line2D([0], [0], marker="$k$", markersize=10, color='gray'))
names.append('k iterations')


plt.legend(custom_lines, names)
plt.xticks(np.log2(nnproc), nnproc)
plt.ylabel('efficiency')
plt.xlabel('number of cores')
plt.title(NAME + ' equation')

# plt.show()
plt.savefig('AAplots/efficiency_' + NAME + '_interval_coll', dpi=300)


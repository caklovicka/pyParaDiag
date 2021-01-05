import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# path = ['heat1/output/000000/result/result.dat', 'heat2/output/000000/result/result.dat', 'heat3/output/000000/result/result.dat']
# path = ['adv1/output/000002/result/result.dat', 'adv2/output/000002/result/result.dat', 'adv3/output/000002/result/result.dat']
path = ['schro1/output/000002/result/result.dat', 'schro2/output/000002/result/result.dat', 'schro3/output/000002/result/result.dat']

# rolling | time_intervals | tot_time | paralpha_iters | tol
heat = [np.loadtxt(path[0], delimiter='|', usecols=[0, 1, 2, 4, 9], skiprows=3),
        np.loadtxt(path[1], delimiter='|', usecols=[0, 1, 2, 4, 9], skiprows=3),
        np.loadtxt(path[2], delimiter='|', usecols=[0, 1, 2, 4, 9], skiprows=3)]

no_runs = len(heat[0]) // 2
speedup = np.ones((no_runs, len(path)))
efficiency = np.ones((no_runs, len(path)))
marks = 10
lw = 2
col = sns.color_palette("bright", len(heat))
custom_lines = []

for run in range(len(heat)):
    for subrun in range(2 * no_runs):
        # if rolling = 1
        if heat[run][subrun, 0] == 1:
            row = int(np.log2(heat[run][subrun, 1])) - 2
            speedup[row, run] /= heat[run][subrun, 2]
            efficiency[row, run] /= heat[run][subrun, 1]
            efficiency[row, run] /= heat[run][subrun, 2]
        else:
            row = int(np.log2(heat[run][subrun, 0])) - 2
            speedup[row, run] *= heat[run][subrun, 2]
            efficiency[row, run] *= heat[run][subrun, 2]

legend = []
proc = range(no_runs)
labels = ['4', '8', '16', '32', '64']
names = ['1e-5', '1e-9', '1e-12']
markers_in_use = set()

# speedup plot
plt.subplot(121)
for run in range(len(heat)):
    plt.plot(proc, speedup[:, run], '--', linewidth=lw, color=col[run])
    custom_lines.append(Line2D([0], [0], linestyle='--', linewidth=lw, color=col[run]))

for run in range(len(heat)):
    for subrun in range(no_runs):
        m = int(heat[run][subrun, 3])
        plt.plot(subrun, speedup[subrun, run], marker="$" + str(m) + "$", markersize=marks, color=col[run])
        markers_in_use.add(int(heat[run][subrun, 3]))

for m in markers_in_use:
    custom_lines.append(Line2D([0], [0], marker="$" + str(m) + "$", markersize=marks, color='gray'))
    names.append(str(m) + ' iter')
plt.legend(custom_lines, names)
plt.xticks(proc, labels)
plt.title('Speedup')
plt.xlabel('nproc')

# efficiency plot
plt.subplot(122)
for run in range(len(heat)):
    plt.plot(proc, efficiency[:, run], '--', linewidth=lw, color=col[run])
    custom_lines.append(Line2D([0], [0], linestyle='--', linewidth=lw, color=col[run]))

for run in range(len(heat)):
    for subrun in range(no_runs):
        m = int(heat[run][subrun, 3])
        plt.plot(subrun, efficiency[subrun, run], marker="$" + str(m) + "$", markersize=marks, color=col[run])
        markers_in_use.add(int(heat[run][subrun, 3]))

plt.legend(custom_lines, names)
plt.xticks(proc, labels)
plt.title('Efficiency')
plt.xlabel('nproc')

plt.show()


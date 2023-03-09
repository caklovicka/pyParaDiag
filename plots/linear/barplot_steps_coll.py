import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

IDX = 2
eq = 'adv'

plt.rcParams["figure.figsize"] = (7,9)

path_heat = ['data/heat1_2.dat', 'data/heat2_5.dat', 'data/heat3_36.dat']
path_adv = ['data/adv1_12.dat', 'data/adv2_8.dat', 'data/adv3_9.dat']

if eq == 'adv':
    NAME = path_adv[IDX]
else:
    NAME = path_heat[IDX]

# nproc | tot_time | iters | comm_time
data = np.loadtxt(NAME, delimiter='|', usecols=[0, 3, 7, 6], skiprows=3)

# sort runs
for i in range(len(data)):
    indices = np.argsort(data[:, 0])
    data = data[indices, :]

col = sns.color_palette("bright", len(data))

comm_time = data[:, 3]
tot_time = data[:, 1]
groups = data[:, 0]
iters = data[:, 2]

labels = []
for i in range(len(groups)):
    labels.append(str(int(groups[i])))

fig, ax = plt.subplots()
ax.bar(labels, comm_time, label = "communication", color='black')
ax.bar(labels, tot_time - comm_time, bottom = comm_time, label = "runtime excluding communication", color=col[IDX])

max_time = max(tot_time)
shift = 0.01 * max_time
for i, total in enumerate(tot_time):
    ax.text(i, total + shift, '{0:g}'.format(round(total, 2)), ha = 'center', weight = 'bold', color = 'black')

for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
          bar.get_height() / 2 + bar.get_y(),
          round(bar.get_height(), 2), ha = 'center',
          color = 'w', weight = 'bold', size = 10)

ax.legend()
ax.set_ylabel('wallclock time [s]')
ax.set_xlabel('total number of cores')
ax.set_ylim([0, max_time + 10 * shift])

plt.tight_layout()
plt.show()
#plt.savefig('strong_plots/Stepparallelstrong' + NAME, dpi=300, bbox_inches='tight')
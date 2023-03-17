import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

IDX = 0

plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams["figure.dpi"] = 150

path_adv = ['data/adv1_11.dat', 'data/adv1_10.dat', 'data/adv2_6.dat', 'data/adv3_7.dat']

# nproc | tot_time | iters | comm_time
data = np.loadtxt(path_adv[IDX], delimiter='|', usecols=[0, 3, 7, 6], skiprows=3)

# sort runs
for i in range(len(data)):
    indices = np.argsort(data[:, 0])
    data = data[indices, :]

col = ['silver'] + sns.color_palette("hls", 3)

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

k = 1
for bar in ax.patches:
    if k == len(data) + 1:
       break
    k += 1
    ax.text(bar.get_x() + bar.get_width() / 2,
          bar.get_height() / 2 + bar.get_y(),
          round(bar.get_height(), 2), ha = 'center',
          color = 'w', weight = 'bold', size = 10)

ax.legend()
ax.set_ylabel('wallclock time [s]', fontsize=15)
ax.set_xlabel('total number of cores', fontsize=15)
#ax.set_ylim([0, 12])

plt.tight_layout()
plt.show()
#plt.savefig('strong_plots/Stepparallelstrong' + NAME, dpi=300, bbox_inches='tight')
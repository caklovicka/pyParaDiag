import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

epsilon = [r'$n_{\rm{step}} = 1$', r'$n_{\rm{step}} = 2$', r'$n_{\rm{step}} = 4$', r'$n_{\rm{step}} = 8$',
          r'$n_{\rm{step}} = 16$', r'$n_{\rm{step}} = 32$']
nstep = [r'$\varepsilon = 1e-1$', r'$\varepsilon = 5e-2$', r'$\varepsilon = 1e-2$', r'$\varepsilon = 8e-3$']

runtimes = np.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                     [1.375, 1.375, 1.625, 1.625],
                     [2, 2, 3, 3],
                     [2, 2, 3, 3],
                     [2, 3, 3, 4]]).transpose()

fig, ax = plt.subplots()
im = ax.imshow(runtimes, cmap='RdYlGn_r')
col = sns.color_palette("bright", 4)

# Show all ticks and label them with the respective list entries
ax.set_yticks(np.arange(len(nstep)), labels=nstep, fontsize=15)
ax.set_xticks(np.arange(len(epsilon)), labels=epsilon, fontsize=15)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(nstep)):
    for j in range(len(epsilon)):
        text = ax.text(j, i, runtimes[i, j], ha="center", va="center", color="black", fontsize=18, weight='book')

for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), col):
    ticklabel.set_color(tickcolor)

fig.tight_layout()
plt.show()
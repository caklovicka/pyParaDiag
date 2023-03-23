import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

alphas = [r'$\alpha$-adaptive', r'IR, $\alpha = 10^{-4}$', r'IR, $\alpha = 10^{-8}$']
equations = [r'heat, $\zeta = 10^{-5}$', r'heat, $\zeta = 10^{-9}$', r'heat, $\zeta = 10^{-12}$', r'advection, $\zeta = 10^{-5}$', r'advection, $\zeta = 10^{-9}$', r'advection, $\zeta = 10^{-12}$']

runtimes = np.array([[2, 2, 42, 6, 9, 155],
                    [10, 737, 700, 2, 6, 97],
                    [806, 1130, 3960, 2, 3, 63]])

fig, ax = plt.subplots()
im = ax.imshow(runtimes, cmap='RdYlGn_r')

# Show all ticks and label them with the respective list entries
ax.set_yticks(np.arange(len(alphas)), labels=alphas, fontsize=12)
ax.set_xticks(np.arange(len(equations)), labels=equations, fontsize=12)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(alphas)):
    for j in range(len(equations)):
        text = ax.text(j, i, runtimes[i, j], ha="center", va="center", color="black", fontsize=18, weight='book')

#ax.set_title("Wallclock time [s]")
fig.tight_layout()
plt.show()
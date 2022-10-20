import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

alphas = ["old, adaptive", "new, 1e-4", "new, 1e-8"]
equations = ["heat, tol = 1e-5", "heat, tol = 1e-9", "heat, tol = 1e-12", "advection, tol = 1e-5", "advection, tol = 1e-9", "advection, tol = 1e-12"]

runtimes = np.array([[2, 2, 42, 6, 9, 155],
                    [10, 737, 700, 2, 6, 97],
                    [806, 1130, 3960, 2, 3, 63]])

fig, ax = plt.subplots()
im = ax.imshow(runtimes, cmap='Reds')

# Show all ticks and label them with the respective list entries
ax.set_yticks(np.arange(len(alphas)), labels=alphas)
ax.set_xticks(np.arange(len(equations)), labels=equations)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(alphas)):
    for j in range(len(equations)):
        text = ax.text(j, i, runtimes[i, j], ha="center", va="center", color="g")

ax.set_title("GMRES iterations")
fig.tight_layout()
plt.show()
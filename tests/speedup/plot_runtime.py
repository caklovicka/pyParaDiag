import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

alphas = ["old, adaptive", "new, 1e-4", "new, 1e-8"]
equations = ["heat, tol = 1e-5", "heat, tol = 1e-9", "heat, tol = 1e-12", "advection, tol = 1e-5", "advection, tol = 1e-9", "advection, tol = 1e-12"]

runtimes = np.array([[0.26107, 0.92291, 4.66083, 1.29374, 5.47151, 40.11753],
                    [0.21547, 16.40066, 29.19395, 0.70112, 3.76745, 32.19536],
                    [9.47420, 26.45528, 146.59290, 0.70069, 1.95702, 17.31436]])


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

ax.set_title("Runtime in seconds")
fig.tight_layout()
plt.show()
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

def u_heat(t, z):
    return np.sin(2 * np.pi * z[0]) * np.sin(2 * np.pi * z[1]) * np.cos(t)

def u_adv(t, z):
    return np.sin(2 * np.pi * (z[0] - t)) * np.sin(2 * np.pi * (z[1] - t))


u = u_adv
T0 = 0
dT = 0.0128

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
dd = 1e-3
X = np.arange(0, 1, dd)
Y = np.arange(0, 1, dd)
X, Y = np.meshgrid(X, Y)

# Plot the surface.
#surf = ax.plot_surface(X, Y, u(T0 + dT, [X, Y]) - u(T0, [X, Y]), cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, u(T0, [X, Y]), cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=-1, vmax=1)

# Customize the z axis.
#ax.set_zlim(-1.01 * dt, 1.01 * dt)
#ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.6, aspect=10, location='left')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

'plotting the a fully serial and interval-parallel speedup'

path = 'heat3_strong/output/000000/result/result.dat'

# nproc | tot_time
eq3 = np.loadtxt(path, delimiter='|', usecols=[0, 3], skiprows=3)

no_runs = len(eq3) - 1
speedup = np.ones(no_runs)

for i in range(no_runs):
    # if seq
    print(speedup)
    if eq3[i, 0] == 1:
        speedup *= eq3[i, 1]
    else:
        row = int(np.log2(eq3[i, 0])) - 2
        speedup[row] /= eq3[i, 1]
plt.plot(speedup)

plt.show()


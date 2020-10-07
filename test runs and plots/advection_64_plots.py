import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

alphas = [3.2654641958779155e-06, 0.0012777840576321798, 0.025276313592296047, 0.11241955700031922]
alphas2 = [3.27e-06, 1.28e-3, 0.0253, 0.112]

# optimal alphas
err = [2.0550924680434268e-08, 2.6811356485696096e-11, 3.84795858282937e-13, 4.181106536838296e-13]
m = [6.530928391755831e-08, 1.6690232361045944e-10, 8.437350941721699e-12, 1.897046510249159e-12]
m2 = [6.5e-8, 1.7e-10, 8.4e-12, 1.9e-12]

# for alpha0
err0 = [2.0550924680434268e-08, 1.4935649005800947e-10, 1.487973911048266e-10, 1.487973911048266e-10, 1.487973911048266e-10]

# for alpha1
err1 = [8.03877202126824e-06, 1.028542651259751e-08, 1.3013173950888462e-11, 5.768807738368252e-13, 6.457587777710884e-13]

# for alpha2
err2 = [0.0001629327132315339, 4.225124969003957e-06, 1.0956440155141085e-07, 2.8415383379965675e-09, 7.336587050817802e-11]

# for alpha3
err3 = [0.0007958032575794416, 0.00010079370965238033, 1.2766276954678801e-05, 1.616940751347773e-06, 2.0479624207592198e-07]


# convert
err = np.log10(np.array(err))
err0 = np.log10(np.array(err0))
err1 = np.log10(np.array(err1))
err2 = np.log10(np.array(err2))
err3 = np.log10(np.array(err3))


n = 7
x = list(range(1, n+1, 1))
# col = sns.color_palette("hls", 3)
col = sns.xkcd_palette(["windows blue", "red", "faded green", "dusty purple"])
linst = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]
marksz = 15
lw = 3
roundoff = np.log10(np.array(m))
mm = [r'$\mathbf{m_1} = }$', r'$\mathbf{m_2} = }$', r'$\mathbf{m_3} = }$', r'$\mathbf{m_4} = }$']

n = 7
for i in range(len(mm)):
    plt.plot(range(n), np.ones(n)*roundoff[i], linestyle=linst[i], color='silver', lw=lw)
    plt.text(0.6, roundoff[i] + 0.1, mm[i] + str(m2[i]), fontsize=marksz + 1, color='silver', weight='bold')


plt.plot(range(1, len(err0)+1, 1), err0, 'X', color=col[0], markersize=marksz, linestyle=linst[0], lw=lw)
plt.plot(range(1, len(err1)+1, 1), err1, 'X', color=col[1], markersize=marksz, linestyle=linst[1], lw=lw)
plt.plot(range(1, len(err2)+1, 1), err2, 'X', color=col[2], markersize=marksz, linestyle=linst[2], lw=lw)
plt.plot(range(1, len(err3)+1, 1), err3, 'X', color=col[3], markersize=marksz, linestyle=linst[3], lw=lw)


n = len(err)
x = list(range(1, n+1, 1))
plt.plot(x, err, 'X', color='black', markersize=marksz, linestyle='solid', lw=lw)

plt.ylim([-13, 0])
plt.xlim([0.5, 5.5])

custom_lines = [Line2D([0], [0], color=col[0], marker='', linestyle=linst[0], lw=lw, markersize=marksz),
                Line2D([0], [0], color=col[1], marker='', linestyle=linst[1], lw=lw, markersize=marksz),
                Line2D([0], [0], color=col[2], marker='', linestyle=linst[2], lw=lw, markersize=marksz),
                Line2D([0], [0], color=col[3], marker='', linestyle=linst[3], lw=lw, markersize=marksz),
                Line2D([0], [0], color='black', linestyle='solid', lw=lw),
                Line2D([0], [0], color='silver', marker='X', markersize=marksz, lw=lw)]

names = [r'$\alpha = $' + str(alphas2[0]), r'$\alpha = $' + str(alphas2[1]), r'$\alpha = $' + str(alphas2[2]), r'$\alpha = $' + str(alphas2[3]), 'optimal', r'$\|u^{(k)}_L - u(T_L)\|_\infty $']
plt.legend(custom_lines, names, prop={'size': 20})
plt.xlabel('Iteration', fontsize=20)
plt.tick_params(labelsize=marksz)

plt.show()
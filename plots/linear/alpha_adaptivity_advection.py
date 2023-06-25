import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

alphas = [6.189239916076504e-07, 0.000556293084447241, 0.01667772593082224, 0.09131737493714501]
alphas2 = [6.19e-07, 0.000556, 0.0167, 0.0913]

# optimal alphas
err = [4.994783647832435e-08, 2.8056047820114626e-11, 6.690062510102548e-13, 5.032046166571533e-13]
m = [1.5844454185155853e-07, 1.7628320580086693e-10, 5.8800059851071836e-12, 1.0738934223493788e-12]
m2 = [1.58e-7, 1.76e-10, 5.88e-12, 1.07e-12]

# for alpha0
err0 = [4.994783647832435e-08, 7.6817049788477e-10, 7.731953599319242e-10, 7.728230113146195e-10, 7.627147308295181e-10, 7.643785657252454e-10]

# for alpha1
err1 = [4.471606188816146e-05, 2.4888949018440783e-08, 1.4165184976373149e-11, 1.1447249940699925e-12, 1.126610593685307e-12]

# for alpha2
err2 = [0.0013622764747797667, 2.3099748669097586e-05, 3.916986562879644e-07, 6.642223660081918e-09, 1.1268636033442073e-10, 2.1811904170870503e-12]

# for alpha3
err3 = [0.008061980523784651, 0.0008090273159575556, 8.118721922573124e-05, 8.147216030485183e-06, 8.17575491834921e-07, 8.204551038559828e-08]


# convert
'''
err = np.log10(np.array(err))
err0 = np.log10(np.array(err0))
err1 = np.log10(np.array(err1))
err2 = np.log10(np.array(err2))
err3 = np.log10(np.array(err3))
'''


n = 7
x = list(range(1, n+1, 1))
# col = sns.color_palette("hls", 3)
col = sns.xkcd_palette(["windows blue", "red", "faded green", "dusty purple"])
linst = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]
marksz = 15
lw = 3
roundoff = np.array(m)#np.log10(np.array(m))
mm = [r'$\mathbf{m_1} = }$', r'$\mathbf{m_2} = }$', r'$\mathbf{m_3} = }$', r'$\mathbf{m_4} = }$']

n = 8
#for i in range(len(mm)):
#    plt.semilogy(range(n), np.ones(n)*roundoff[i], linestyle=linst[i], color='silver', lw=lw)
#    plt.text(0.6, roundoff[i] * 1.3, mm[i] + str(m2[i]), fontsize=marksz + 1, color='silver', weight='bold')


plt.semilogy(range(1, len(err0)+1, 1), err0, 'X', color=col[0], markersize=marksz, linestyle=linst[0], lw=lw)
plt.semilogy(range(1, len(err1)+1, 1), err1, 'X', color=col[1], markersize=marksz, linestyle=linst[1], lw=lw)
plt.semilogy(range(1, len(err2)+1, 1), err2, 'X', color=col[2], markersize=marksz, linestyle=linst[2], lw=lw)
plt.semilogy(range(1, len(err3)+1, 1), err3, 'X', color=col[3], markersize=marksz, linestyle=linst[3], lw=lw)


n = len(err)
x = list(range(1, n+1, 1))
plt.semilogy(x, err, 'X', color='black', markersize=marksz, linestyle='solid', lw=lw)

plt.ylim([10**(-13), 1])
plt.xlim([0.5, 6.5])

custom_lines = [Line2D([0], [0], color=col[0], marker='', linestyle=linst[0], lw=lw, markersize=marksz),
                Line2D([0], [0], color=col[1], marker='', linestyle=linst[1], lw=lw, markersize=marksz),
                Line2D([0], [0], color=col[2], marker='', linestyle=linst[2], lw=lw, markersize=marksz),
                Line2D([0], [0], color=col[3], marker='', linestyle=linst[3], lw=lw, markersize=marksz),
                Line2D([0], [0], color='black', linestyle='solid', lw=lw),
                Line2D([0], [0], color='silver', marker='X', markersize=marksz, lw=lw)]

names = [r'$\alpha = {:.2e}$'.format(alphas2[0]), r'$\alpha = {:.2e}$'.format(alphas2[1]), r'$\alpha = {:.2e}$'.format(alphas2[2]), r'$\alpha = {:.2e}$'.format(alphas2[3]), 'adaptive']
plt.legend(custom_lines, names, prop={'size': 20})
plt.xlabel('iteration', fontsize=25)
plt.ylabel(r'$\|u^{(k)}_L - u(T_L)\|_\infty $', fontsize=25)
plt.tick_params(labelsize=marksz)

plt.show()
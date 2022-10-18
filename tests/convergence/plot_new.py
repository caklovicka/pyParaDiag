import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure


heat1 = [[3.5283668211594232e-06, 3.528018905241126e-06],
         [3.7929742862097525e-06],
         [0.003470391845988834, 4.168636823986974e-06, 3.6024428434755617e-06]]
heat2 = [[5.392192252458673e-10, 7.232879850604945e-10],
         [8.113322387159215e-09, 4.5062065190418946e-10, 7.477007902140678e-10],
         [1.6429238499984317e-05, 7.468908528758617e-08, 2.6705228695081086e-08, 2.5451528884445906e-08, 2.902767337609191e-08]]
heat3 = [[1.8554302635954418e-10, 4.797273970639484e-13],
         [6.03445986221889e-11, 9.455769503887314e-13, 1.6512347047695585e-12, 2.1753709944505317e-12, 2.9878322038712213e-12],
         [5.146307486827638e-08, 7.136438112721621e-10, 1.942123684095876e-10, 3.997697728408412e-10, 5.839829730902579e-10]]
adv1 = [[1.3116282530756251e-05, 6.839703235339556e-06],
        [6.8408533447921776e-06, 6.840323967360899e-06],
        [5.770640713361787e-06, 6.840327481882902e-06]]
adv2 = [[6.2800316187328065e-06, 4.3052872378300765e-10, 5.422776672148188e-10],
        [1.4378137370843578e-08, 5.423436144624842e-10, 5.422263749110812e-10],
        [0.010519865125722117, 5.430784710848402e-10, 5.42680012038967e-10]]
adv3 = [[6.279675234255322e-06, 6.283396025708048e-10, 5.163647287531603e-13, 5.276890036043369e-13],
        [6.998659087895124e-10, 5.278000259397244e-13, ],
        [1.0598943743584827e-06, 5.280220705287757e-13, 5.27689003604337e-13]]

alphas = [1e-4, 1e-8, 1e-12]
tol = [1e-5, 1e-9, 1e-12]
n = len(tol)

plt.rcParams["figure.figsize"] = 10, 10
plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.5)

col = sns.color_palette("hls", n)
linst = ['dotted', 'dashed', 'dashdot']
marksz = 13
lw = 2
tx = 3.3

# HEAT
for i in range(n):
    if i == 0:
        plt.subplot(231)
    elif i == 1:
        plt.subplot(232)
    else:
        plt.subplot(233)

    x = range(1, len(heat1[i]) + 1, 1)
    plt.semilogy(x, heat1[i], marker='X', color=col[0], linestyle=linst[0], markersize=marksz, lw=lw)

    x = range(1, len(heat2[i]) + 1, 1)
    plt.semilogy(x, heat2[i], marker='X', color=col[1], linestyle=linst[1], markersize=marksz, lw=lw)

    x = range(1, len(heat3[i]) + 1, 1)
    plt.semilogy(x, heat3[i], marker='X', color=col[2], linestyle=linst[2], markersize=marksz, lw=lw)

    l = max(len(heat1[i]), len(heat2[i]), len(heat3[i]))

    for j in range(n):
        x = range(0, l + 2, 1)
        plt.semilogy(x, np.ones(l + 2) * tol[j], linestyle=linst[j], color='silver', lw=lw)
        plt.text(tx, np.log10(tol[j]) + 0.3, str(tol[j]), fontsize=marksz + 1, weight='bold', color='silver')

    plt.legend(tol)
    plt.title(r'$Heat, \alpha = $' + str(alphas[i]))
    plt.xlabel('iteration')

# ADVECTION
for i in range(n):
    if i == 0:
        plt.subplot(234)
    elif i == 1:
        plt.subplot(235)
    else:
        plt.subplot(236)

    x = range(1, len(adv1[i]) + 1, 1)
    plt.semilogy(x, adv1[i], marker='X', color=col[0], linestyle=linst[0], markersize=marksz, lw=lw)

    x = range(1, len(adv2[i]) + 1, 1)
    plt.semilogy(x, adv2[i], marker='X', color=col[1], linestyle=linst[1], markersize=marksz, lw=lw)

    x = range(1, len(adv3[i]) + 1, 1)
    plt.semilogy(x, adv3[i], marker='X', color=col[2], linestyle=linst[2], markersize=marksz, lw=lw)

    l = max(len(adv1[i]), len(adv2[i]), len(adv3[i]))

    for j in range(n):
        x = range(0, l + 2, 1)
        plt.semilogy(x, np.ones(l + 2) * tol[j], linestyle=linst[j], color='silver', lw=lw)
        plt.text(tx, np.log10(tol[j]) + 0.3, str(tol[j]), fontsize=marksz + 1, weight='bold', color='silver')

    plt.legend(tol)
    plt.title(r'$Advection, \alpha = $' + str(alphas[i]))
    plt.xlabel('iteration')

plt.show()
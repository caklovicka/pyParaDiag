import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure


heat1 = [[3.528366757210577e-06],
         [3.793144030872516e-06, 3.5282190952168335e-06],
         [0.0034703870125102902, 4.0979869421201625e-06]]
heat2 = [[5.395095491498496e-10, 7.232603405072011e-10],
         [1.103899610423694e-08, 7.21403492498496e-10],
         [3.256885926726096e-05, 4.755158878193872e-10, 1.345879074854794e-09, 1.0571619135646415e-09, 1.5148046150770258e-09]]
heat3 = [[1.8553292250129867e-10, 4.802824944587035e-13],
         [7.39653967066683e-11, 4.760636401300593e-13],
         [1.2172806246272216e-07, 3.5129678401591842e-12, 3.5380587348760923e-12, 1.5093329364116006e-11, 1.4553802607509377e-11]]
adv1 = [[1.311628285571853e-05],
        [6.841433189631454e-06],
        [6.067485416653954e-06]]
adv2 = [[6.28003140723532e-06, 4.305363843215567e-10],
        [6.007413016856248e-10],
        [3.2138146758344213e-07, 5.422396975873793e-10]]
adv3 = [[6.279675256681827e-06, 6.283399356377139e-10, 5.163647287531603e-13],
        [8.845543969152299e-10, 5.279110491613059e-13],
        [3.4540300353407576e-06, 5.28022070910768e-13]]

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
        plt.ylabel(r'$|u_L^{(k)} - u(T_L)|_\infty$', fontsize=15)
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
        plt.semilogy(x, np.ones(l + 2) * tol[j], linestyle=linst[j], color=col[j], lw=lw//2)
        plt.text(tx, np.log10(tol[j]) + 0.3, str(tol[j]), fontsize=marksz + 1, weight='bold', color='silver')

    plt.legend(tol)
    plt.title(r'$Heat, \alpha = $' + str(alphas[i]))
    plt.xlabel('iteration', fontsize=10)

# ADVECTION
for i in range(n):
    if i == 0:
        plt.subplot(234)
        plt.ylabel(r'$|u_L^{(k)} - u(T_L)|_\infty$', fontsize=15)
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
        plt.semilogy(x, np.ones(l + 2) * tol[j], linestyle=linst[j], color=col[j], lw=lw//2)
        plt.text(tx, np.log10(tol[j]) + 0.3, str(tol[j]), fontsize=marksz + 1, weight='bold', color='silver')

    plt.legend(tol)
    plt.title(r'$Advection, \alpha = $' + str(alphas[i]))
    plt.xlabel('iteration', fontsize=10)

plt.tight_layout()
plt.show()
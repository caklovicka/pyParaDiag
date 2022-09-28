import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure


heat1 = [[0.013772124389615103, 0.00018647995734422275, 3.732610349806093e-06, 3.8124624628377765e-06],
         [4.867184621559793e-06, 4.678629395682421e-06],
         [3.5283698719412726e-06, .528527980023455e-06]]
heat2 = [[0.018857793680243002, 0.0001598682336956736, 3.040997871201933e-07, 2.664807471408379e-07, 3.1992375082446e-07, 1.3732782022657375e-07, 2.066437879699734e-07, 2.572869480799156e-07, 3.190100551497821e-07, 2.973422612573273e-07],
         [2.3664169716268994e-06, 8.536704519945822e-07, 8.383633787356184e-10, 9.289375935139788e-10],
         [7.835223583143458e-10, 1.4785969115749253e-09, 7.141501834340154e-10, 7.241242050426422e-10]]
heat3 = [[1.2148487065219247e-05, 3.2798408226617495e-05, 1.2932863380327583e-07, 1.8675002566582313e-07, 9.743619699964512e-08, 1.2972029397873008e-07, 1.6541621805643558e-07, 1.3469510040664545e-07, 1.8669191792497486e-07, 8.689508124160028e-08],
         [4.247626518298441e-06, 9.177802295923556e-09, 2.8554825172248864e-11, 2.3611224087005673e-11, 2.254474384955074e-11, 1.777178404438473e-11, 2.0956569812824455e-11, 2.3985036179396957e-11, 2.3588242470395926e-11, 2.1155299734232358e-11],
         [8.083546055643547e-10, 1.0070252409732995e-09, 2.653433035153798e-13, 5.053735208093723e-13]]
adv1 = [[5.770640713361787e-06, 6.841788968311667e-06],
        [6.8408533447921776e-06, 6.840325012858019e-06],
        [1.3116282530756251e-05, 6.839703235339556e-06]]
adv2 = [[1.4001733326099655e-05, 9.32678601765038e-08, 8.83782380629677e-10],
        [1.4378137370843578e-08, 5.461013019402392e-10, 5.42240252698889e-10],
        [6.2800316187328065e-06, 4.304900880241879e-10, 5.434812599975766e-10]]
adv3 = [[2.5717865543374015e-05, 3.4727629710754246e-07, 5.566636045016667e-10, 3.96378818656759e-10, 3.995404007639536e-10, 3.7024594501389174e-10, 4.0906222853465124e-10, 3.396575243286293e-10, 3.982910667943429e-10, 3.6351432974868203e-10],
        [6.9069427599332565e-09, 7.300625612010511e-11, 5.991873700350254e-13, 5.67546010191297e-13],
        [6.279678869736127e-06, 6.365046267984328e-10, 1.38427048341519e-11, 5.332401187276271e-13, 5.27689003604337e-13]]

alphas = [1e-12, 1e-8, 1e-4]
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
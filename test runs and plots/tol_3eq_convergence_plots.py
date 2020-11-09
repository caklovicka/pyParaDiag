import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

tol = [1e-5, 1e-9, 1e-12]

# heat
herr_abs1 = [6.244210700945274e-06, 6.3223745696561195e-06]
herr_abs2 = [3.7758531931812573e-10, 7.274646553940157e-10]
herr_abs3 = [1.2591146446553889e-08, 2.950075255740583e-11, 1.1020229980500296e-12, 5.830397893467526e-13]
heat_err_abs = [herr_abs1, herr_abs2, herr_abs3]

hcerr1 = [0.0049893470243037585, 7.816386871084546e-08]
hcerr2 = [0.0049958347258076685, 7.312931371572517e-10]
hcerr3 = [0.004995846875994525, 1.2139308003256133e-08, 2.8718027955676462e-11, 3.8891112552619234e-13]
heat_cerr = [hcerr1, hcerr2, hcerr3]

hm1 = [0.005362537221768715, 0.0012418109342038387]
hm2 = [3.994785610141585e-05, 7.984361929631986e-07]
hm3 = [9.809076291256445e-07, 3.072148244679859e-09, 1.7192902351257633e-10, 4.067267258024639e-11]
m_heat = [[1.5243813571647907e-06, 1.2148840277250229e-08], [1.2040996413616275e-07, 1.0788175703047926e-09], [2.770860129567921e-09, 2.1052384174382418e-11, 1.83503616672295e-12]]


# schrodinger
serr_abs1 = [0.0002248108053902203, 2.318475442350056e-05, 6.204178977392176e-06]
serr_abs2 = [3.5764516545302237e-06, 3.487043981972412e-08, 1.9295007081495433e-09, 1.2458760123108672e-09, 4.920879400707415e-10]
serr_abs3 = []
schro_err_abs = []

schro_cerr = []
m_schro = []

# advection
adv_err_abs = [[1.3459007116962887e-07], [5.784215829501536e-10, 3.003577408612999e-10], [1.9441193724689305e-10, 2.3756729368312857e-12, 6.906144555656683e-13, 8.195697777274086e-13]]
adv_cerr = [[0.001507821492504273], [3.015986656962344e-05, 8.749800883833814e-10], [1.5081588814469526e-06, 1.9678125795508095e-10, 1.6928680679484387e-12, 1.4599432773820809e-13]]
#adv_final_abs = [1.3910147964382724e-07,  2.969837709088097e-10,  7.925882172798993e-13]
m_adv = [[1.9591829064620917e-08], [2.7708382519525807e-09, 2.1052134843232613e-11], [6.19578268602266e-10, 9.954939939795315e-12, 1.2618562348951718e-12, 4.492576642250953e-13]]



# covnert
n = len(tol)
for i in range(n):
    heat_err_abs[i] = np.log10(np.array(heat_err_abs[i]))
    heat_cerr[i] = np.log10(np.array(heat_cerr[i]))
    m_heat[i] = np.log10(np.array(m_heat[i]))
    #heat_final_abs[i] = np.log10(np.array(heat_final_abs[i]))
    schro_err_abs[i] = np.log10(np.array(schro_err_abs[i]))
    schro_cerr[i] = np.log10(np.array(schro_cerr[i]))
    m_schro[i] = np.log10(np.array(m_schro[i]))
    #schro_final_abs[i] = np.log10(np.array(schro_final_abs[i]))
    adv_err_abs[i] = np.log10(np.array(adv_err_abs[i]))
    adv_cerr[i] = np.log10(np.array(adv_cerr[i]))
    m_adv[i] = np.log10(np.array(m_adv[i]))
    #adv_final_abs[i] = np.log10(np.array(adv_final_abs[i]))


col = sns.color_palette("hls", n)
linst = ['dotted', 'dashed', 'dashdot']
marksz = 13
lw = 2
custom_lines1 = [Line2D([0], [0], color='silver', marker=8, markersize=marksz, lw=lw),
                Line2D([0], [0], color='silver', marker=9, markersize=marksz, lw=lw)]

custom_lines2 = [Line2D([0], [0], color='silver', marker='X', markersize=marksz, lw=lw)]

names1 = [r'$\|u^{(k-1)}_L - u^{(k)}_L\|_\infty$', r'$m_k$']
names2 = [r'$\|u(T_L) - u^{(k)}_L\|_\infty$']

# HEAT

plt.subplot(231)
l = 5
tx = 3.8
d = 0.05
ts = 20

leg = []
for i in range(n):
    x = range(0, l + 2, 1)
    plt.plot(x, np.ones(l + 2) * np.log10(tol[i]), linestyle=linst[i], color='silver', lw=lw)
    plt.text(tx, np.log10(tol[i]) + 0.3, str(tol[i]), fontsize=marksz + 1, weight='bold', color='silver')

for i in range(n):
    l = heat_cerr[i].shape
    l = l[0]
    x = range(1, l+1, 1)
    plt.plot(x, heat_cerr[i], marker=8, color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

for i in range(n):
    l = len(m_heat[i])
    x = range(1, l+1, 1)
    plt.plot(x, m_heat[i], marker=9, color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

plt.legend(custom_lines1, names1, prop={'size': marksz})
plt.xticks([])
plt.ylim([-14, -1])
plt.xlim([0, 5])
plt.title('Heat', fontsize=ts)

plt.subplot(234)

l = 5
for i in range(n):
    x = range(0, l + 2, 1)
    plt.plot(x, np.ones(l + 2) * np.log10(tol[i]), linestyle=linst[i], color='silver', lw=lw)
    plt.text(tx, np.log10(tol[i]) + 0.3, str(tol[i]), fontsize=marksz + 1, weight='bold', color='silver')

for i in range(n):
    l = heat_err_abs[i].shape
    l = l[0]
    x = range(1, l+1, 1)
    plt.plot(x, heat_err_abs[i], 'X', color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

plt.legend(custom_lines2, names2, prop={'size': marksz})
plt.ylim([-14, -2])
plt.xlim([0, 5])


#SCHRODINGER

plt.subplot(232)

for i in range(n):
    x = range(0, l+3, 1)
    print(l)
    plt.plot(x, np.ones(l + 3) * np.log10(tol[i]), linestyle=linst[i], color='silver', lw=lw)
    plt.text(tx, np.log10(tol[i]) + 0.3, str(tol[i]), fontsize=marksz + 1, weight='bold', color='silver')

for i in range(n):
    l = schro_cerr[i].shape
    l = l[0]
    x = range(1, l + 1, 1)
    plt.plot(x, schro_cerr[i], marker=8, color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

for i in range(n):
    l = len(m_schro[i])
    x = range(1, l+1, 1)
    plt.plot(x, m_schro[i], marker=9, color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

plt.yticks([])
plt.legend(custom_lines1, names1, prop={'size': marksz})
plt.ylim([-14, -2])
plt.xlim([0, 5])
plt.xticks([])
plt.title('Schrodinger', fontsize=ts)

plt.subplot(235)
l = 5
for i in range(n):
    x = range(0, l + 2, 1)
    plt.plot(x, np.ones(l + 2) * np.log10(tol[i]), linestyle=linst[i], color='silver', lw=lw)
    plt.text(tx, np.log10(tol[i]) + 0.3, str(tol[i]), fontsize=marksz + 1, weight='bold', color='silver')

for i in range(n):
    l = schro_err_abs[i].shape
    l = l[0]
    x = range(1, l+1, 1)
    plt.plot(x, schro_err_abs[i], 'X', color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

plt.legend(custom_lines2, names2, prop={'size': marksz})
plt.yticks([])
plt.ylim([-14, -2])
plt.xlim([0, 5])
plt.xlabel('Iteration', fontsize=ts)

# ADVECTION

plt.subplot(233)

for i in range(n):
    x = range(0, l + 2, 1)
    plt.plot(x, np.ones(l + 2) * np.log10(tol[i]), linestyle=linst[i], color='silver', lw=lw)
    plt.text(tx, np.log10(tol[i]) + 0.3, str(tol[i]), fontsize=marksz + 1, weight='bold', color='silver')

for i in range(n):
    l = len(m_adv[i])
    x = range(1, l+1, 1)
    plt.plot(x, m_adv[i], marker=9, color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

for i in range(n):
    l = adv_cerr[i].shape
    l = l[0]
    x = range(1, l + 1, 1)
    plt.plot(x, adv_cerr[i], marker=8, color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

plt.yticks([])
plt.xticks([])

plt.legend(custom_lines1, names1, prop={'size': marksz})
plt.ylim([-14, -2])
plt.xlim([0, 5])
plt.title('Advection', fontsize=ts)

plt.subplot(236)
l = 5
for i in range(n):
    x = range(0, l + 2, 1)
    plt.plot(x, np.ones(l + 2) * np.log10(tol[i]), linestyle=linst[i], color='silver', lw=lw)
    plt.text(tx, np.log10(tol[i]) + 0.3, str(tol[i]), fontsize=marksz + 1, weight='bold', color='silver')

for i in range(n):
    l = adv_err_abs[i].shape
    l = l[0]
    x = range(1, l+1, 1)
    plt.plot(x, adv_err_abs[i], 'X', color=col[i], linestyle=linst[i], markersize=marksz, lw=lw)

plt.legend(custom_lines2, names2, prop={'size': marksz})
plt.yticks([])
plt.ylim([-14, -2])
plt.xlim([0, 5])

plt.show()
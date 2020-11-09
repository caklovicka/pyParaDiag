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
m_heat = [hm1, hm2, hm3]

# schrodinger
serr_abs1 = [0.0002248108053902203, 2.318475442350056e-05, 6.204178977392176e-06]
serr_abs2 = [3.5764516545302237e-06, 3.487043981972412e-08, 1.9295007081495433e-09, 1.2458760123108672e-09, 4.920879400707415e-10]
serr_abs3 = [1.2900475679222303e-07, 2.3891139379956475e-10, 7.435508431828706e-12, 1.303986622545877e-12, 3.5214948034722806e-13, 4.5174597059667084e-13]
schro_err_abs = [serr_abs1, serr_abs2, serr_abs3]

scerr1 = [0.01628897201662116, 0.00019962062387213209, 1.8648568788859343e-05]
scerr2 = [0.02000166534300019, 3.6112239797855634e-06, 3.674674718999039e-08, 3.147059657577867e-09, 7.924272096513267e-10]
scerr3 = [0.019998264486022547, 1.2924251815992158e-07, 2.460959646963547e-10, 8.51267887025897e-12, 1.1823641232118227e-12, 3.9870466538598083e-13]
schro_cerr = [scerr1, scerr2, scerr3]

sm1 = [0.00011313012069541056, 1.7016975497574466e-05, 6.599855622126867e-06]
sm2 = [1.7888519512298416e-06, 3.3835811463711835e-08, 4.6534783211700125e-09, 1.7257514397150021e-09, 1.050941444510635e-09]
sm3 = [6.366357860761558e-08, 2.271703089271348e-10, 1.3570061919862896e-11, 3.31662950316989e-12, 1.639662728989387e-12, 1.1528778824109547e-12]
m_schro = [sm1, sm2, sm3]

# advection
aerr_abs1 = [0.0001849426377156238, 1.3387618075669325e-06]
aerr_abs2 = [7.943738135340883e-06, 6.327881868495474e-08, 4.744755782918161e-09, 5.032868028532889e-10, 9.279997906344022e-10]
aerr_abs3 = [3.668357907912658e-07, 6.288909397392169e-10, 1.9092482687573266e-11, 3.0030631309815196e-12, 9.890424748069008e-13, 9.20818976899557e-13, 4.3570703173158554e-13]
adv_err_abs = [aerr_abs1, aerr_abs2, aerr_abs3]

acerr1 = [0.06297403743748914, 0.00018500808011900904]
acerr2 = [0.06279840056322794, 8.00655470667433e-06, 6.791945905693098e-08, 5.2014553775237005e-09, 1.3147176680661232e-09]
acerr3 = [0.06279088347534906, 3.6745925224845877e-07, 6.478357295058856e-10, 2.1600832234014433e-11, 3.4597880116393753e-12, 1.2951029138008607e-12, 7.765038612106423e-13]
adv_cerr = [acerr1, acerr2, acerr3]

am1 = [5.656626558145889e-05, 4.254379843106302e-06]
am2 = [2.52980447241819e-06, 4.023744851835272e-08, 5.0746041821264024e-09, 1.8021379926202465e-09, 1.0739423711438422e-09]
am3 = [ 1.1684456034797735e-07, 3.9940431103533617e-10, 2.3351504832573883e-11, 5.646325817824972e-12, 2.776460290017992e-12, 1.9469474605255454e-12, 1.630367895436689e-12]
m_adv = [am1, am2, am3]


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
plt.ylim([-14, 0])
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
plt.ylim([-14, 0])
plt.xlim([0, 5])


#SCHRODINGER

plt.subplot(232)
l = 6
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
plt.ylim([-14, 0])
plt.xlim([0, 7])
plt.xticks([])
plt.title('Schrodinger', fontsize=ts)

plt.subplot(235)
l = 6
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
plt.ylim([-14, 0])
plt.xlim([0, 7])
plt.xlabel('Iteration', fontsize=ts)

# ADVECTION

plt.subplot(233)
l = 9
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
plt.ylim([-14, 0])
plt.xlim([0, 8])
plt.title('Advection', fontsize=ts)

plt.subplot(236)
l = 9
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
plt.ylim([-14, 0])
plt.xlim([0, 8])

plt.show()
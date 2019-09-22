import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import one_simulation_morememo

alpha_reals = []

#M = 3
alphas = np.array([10**(np.log10(0.02) + 0.1 * ni) for ni in range(10)])
M = 3
enes = []
for a in alphas:
    N = int(2**M / a)
    if N % 2 != 0:
        N += 1
    enes.append(N)
    alpha_reals.append(2**M / N)

#M = 5
alphas = np.array([10**(np.log10(0.2) + 0.1 * ni) for ni in range(8)])
M = 5
enes5 = []
for a in alphas:
    N = int(2**M / a)
    if N % 2 != 0:
        N += 1
    enes5.append(N)
    alpha_reals.append(2**M / N)

M = 8
alphas = np.array([10**(np.log10(2) + 0.1 * ni) for ni in range(8)])
enes8 = []
for a in alphas:
    N = int(2**M / a)
    if N % 2 != 0:
        N += 1
    enes8.append(N)
    alpha_reals.append(2**M / N)

enes.extend(enes5)
enes.extend(enes8)

print(np.array(enes))
print(np.array(alpha_reals))

G_S = []
G_M = []
HN_S = []
HN_M = []

numponderas = 100
imprimecada = 10
cont = 0
for N in enes:
    cont += 1
    if cont < 11:
        M = 3
        T = 10000
    if (cont >= 11 and cont < 19):
        M = 5
        T = 7500
    if cont >= 19:
        M = 8
        T = 5000

    mp = M + 1
    print('N = {}, {}/{}'.format(N, cont, len(enes)))
    print('M = ', M)
    print('alpha = ', alpha_reals[cont-1])

    gs = 0
    gm = 0
    hs = 0
    hm = 0
    for i in range(numponderas):
        t, H, HM, gain_s, gain_m = one_simulation_morememo(N, M, mp, T)
        gs += gain_s
        gm += gain_m
        hs += H
        hm += HM
        if (i+1) % imprimecada == 0:
            print('     Ponderacion nº {}/{}'.format(i + 1, numponderas))
            print('         Average_gs = {:.3f}'.format(gs / (i + 1)))
            print('         Average_gm = {:.3f}'.format(gm / (i + 1)))
            print('         Average_Hs/N = {:.3f}'.format(hs / (i + 1)))
            print('         Average_Hm/N = {:.3f}'.format(hm / (i + 1)))
    G_S.append(gs / numponderas)
    G_M.append(gm / numponderas)
    HN_S.append(hs/ numponderas)
    HN_M.append(hm / numponderas)
    print(' ')

file_name = "alpha__gains_infos_mprima_m_emes358" + str(numponderas) + "numponderas"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(G_S)):
        file.write('{} {} {} {} {}\n'.format(alpha_reals[i], G_S[i], G_M[i], HN_S[i], HN_M[i]))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('H/N')
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.plot(alpha_reals,
        HN_S,
        marker='o',
        mec='k',
        mfc='w',
        ls='-',
        label='M players')
ax.plot(alpha_reals,
        HN_M,
        marker='o',
        mec='k',
        mfc='w',
        ls='-',
        label="M'=M+1 player")
ax.legend(loc=0)
plt.show()


fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('average gain per agent')
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.plot(alpha_reals,
        G_S,
        marker='o',
        mec='k',
        mfc='w',
        ls='-',
        label='M players')
ax.plot(alpha_reals,
        G_M,
        marker='o',
        mec='k',
        mfc='w',
        ls='-',
        label="M' player")
ax.axhline(y=0, color='k', linestyle='--')
ax.legend(loc=0)
plt.show()
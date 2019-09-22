import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import GameSimulationInfo

emes = [(2 + i) for i in range(9)]
enes = [71, 85, 101]

alpha = []
H = []
H2 = []
numponderas = 100
imprimecada = 10
for N in enes:
    for M in emes:
        alpha_n = 2**M / N
        alpha.append(alpha_n)
        print(' N = {}        ({}/{})'.format(N, enes.index(N) + 1, len(enes)))
        print(' M = {}               ({}/{})'.format(M,
                                                     emes.index(M) + 1,
                                                     len(emes)))
        print(' --> α = {:.3f} <--'.format(alpha_n))
        h = 0
        h2 = 0
        for pond in range(numponderas):
            info,info2 = GameSimulationInfo(N, M, T= 2* N * 2**M)
            h += info
            h2 += info2
            if (pond + 1) % imprimecada == 0:
                print('       -Ponderacion nº {}/{}'.format(
                    pond + 1, numponderas))
                print('             H = {}'.format(h / (pond + 1)))
                print('             H2= {}'.format(h2 / (pond + 1)))
        H.append(h / numponderas)
        H2.append(h2/numponderas)

H = [x for _, x in sorted(zip(alpha, H))]
H2 = [x for _, x in sorted(zip(alpha, H2))]
alpha.sort()

file_name = 'HvsALPHA_' + '101,121,145N_' + str(numponderas) + 'numponds'
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(alpha)):
        file.write('{} {} {}\n'.format(alpha[i], H[i], H2[i]))

fig, ax = plt.subplots()
ax.set(xscale='log', xlabel=r'$\alpha$', ylabel=r'$\theta^2$')
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(y), 0)))).format(y)))
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.tick_params(which='both',
               labelbottom=True,
               labeltop=False,
               labelleft=True,
               labelright=False,
               bottom=True,
               top=True,
               left=True,
               right=True)
ax.plot(alpha,
        H,
        color='k',
        alpha=0.8,
        marker='o',
        ms=5,
        mec='navy',
        mfc='w',
        ls='')
ax.plot(alpha,
        H2,
        color='r',
        alpha=0.8,
        marker='o',
        ms=5,
        mec='navy',
        mfc='w',
        ls='')
ax.axvline(x=0.337, color='k', alpha=0.6, linestyle='--')
plt.show()
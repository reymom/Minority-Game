import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import GameSimulation

S = 2
emes = [(3 + i) for i in range(13)]
#enes = [101, 143] para el de diferentes S, tambien lo ordenaste al reves, primero bucle emes
enes = [201, 401, 601, 801]
mecs = ['indianred', 'forestgreen', 'navy', 'darkgoldenrod', 'olive']
mark = ['o', 's', '*', '^', '+']

fig, ax = plt.subplots()
ax.set(xscale='log', yscale='log', xlabel=r'$\alpha$', ylabel=r'$\sigma^2/N$')
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

numponderas = 100
imprimecada = 1
for N in enes:
    alpha = []  # X
    sigma = []  # Y
    print(' N = {}               ({}/{})'.format(N,
                                                 enes.index(N) + 1, len(enes)))
    for M in emes:
        alpha_n = 2**M / N
        alpha.append(alpha_n)
        print('     M = {}               ({}/{})'.format(
            M,
            emes.index(M) + 1, len(emes)))
        print('     α = {:.3f}'.format(alpha_n))
        s = 0
        for pond in range(numponderas):
            A = GameSimulation(N, S, M, T=100000)
            s += np.var(A) / N
            if (pond + 1) % imprimecada == 0:
                print('        -Ponderacion nº {}/{}'.format(
                    pond + 1, numponderas))
                print('             σ²/N = {}'.format(s / (pond + 1)))
        sigma.append(s / numponderas)

    file_name = 'VARvsALPHA_' + str(N) + 'N_' + str(numponderas) + 'numponds'
    with open('Data/{}.dat'.format(file_name), 'w') as file:
        for i in range(len(sigma)):
            file.write('{} {}\n'.format(alpha[i], sigma[i]))

    label = 'N = ' + str(N)
    ax.plot(alpha,
            sigma,
            color='k',
            alpha=0.8,
            marker=mark[enes.index(N)],
            ms=5,
            mec=mecs[enes.index(N)],
            mfc='w',
            ls='',
            label=label)

ax.axhline(y=1, xmin=0, xmax=1, color='k', alpha=0.8, linestyle='--')
ax.legend(loc=0)
plt.show()

#sigma = [x for _, x in sorted(zip(alpha, sigma))]
#alpha.sort()

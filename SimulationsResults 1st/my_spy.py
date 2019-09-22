import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import GameSimulationSpy

emes = [(3 + i) for i in range(8)]

alpha = []
GS = []
GSPY = []
numponderas = 10
imprimecada = 10

for N, NB in zip([134, 100], [40, 30]):
    for M in emes:
        alpha.append(2**M / (N + 1))
        print(' N = {}'.format(N))
        print('     M = {}               ({}/{})'.format(
            M,
            emes.index(M) + 1, len(emes)))
        print('     α = {:.3f}'.format(alpha[emes.index(M)]))

        gs = 0
        gspy = 0
        for i in range(numponderas):
            gain_s, gain_spy = one_simulation_spy(N, M, NB, T=10000)
            gs += gain_s
            gspy += gain_spy
            if (i + 1) % imprimecada == 0:
                print('         Ponderacion nº {}/{}'.format(
                    i + 1, numponderas))
                print('             <Gs> = {:.3f}'.format(gs / (i + 1)))
                print('             <Gspy> = {:.3f}'.format(gspy / (i + 1)))
        GS.append(gs / numponderas)
        GSPY.append(gspy / numponderas)

GS = [x for _, x in sorted(zip(alpha, GS))]
GSPY = [x for _, x in sorted(zip(alpha, GSPY))]
alpha.sort()

file_name = "alpha__vs__gainS_gainSPY_" + str(NB) + 'NB_' + str(N)+ 'N_' +str(numponderas) + "ponds"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(GS)):
        file.write('{} {} {}\n'.format(alpha[i], GS[i], GSPY[i]))

#------FIGURE------#
fig, ax = plt.subplots(figsize=(8, 6))

ax.tick_params(which='both',
               labelbottom=True,
               labeltop=False,
               labelleft=True,
               labelright=False,
               bottom=True,
               top=True,
               left=True,
               right=False)
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))

ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Average gain per agent')
ax.plot(alpha, GS, marker='o', mec='k', mfc='w', ls='-', label='Normal agents')
ax.plot(alpha, GSPY, marker='o', mec='k', mfc='w', ls='-', label='Spy')
ax.axhline(y=0, color='k', linestyle='--')
ax.legend(loc=0)

plt.show()
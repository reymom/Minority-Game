import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import one_simulation_spy

alpha = 0.15
N = 854
M = int(np.log2(alpha * (N + 1)))  #7

N = 100
M = 4

GS = []
GSPY = []
numponderas = 100
imprimecada = 10

Nspied = [(2 * i + 1) for i in range(5)]
Nspied.extend([(8 * i + 11) for i in range(10)])

Nspied = [99]

for NB in Nspied:
    print('NB = {}         ({}/{})'.format(NB,
                                           Nspied.index(NB) + 1, len(Nspied)))
    gs = 0
    gspy = 0
    for i in range(numponderas):
        gain_s, gain_spy = one_simulation_spy(N, M, NB, T=5000)
        gs += gain_s
        gspy += gain_spy
        if (i + 1) % imprimecada == 0:
            print('         Ponderacion nº {}/{}'.format(i + 1, numponderas))
            print('             <Gs> = {:.3f}'.format(gs / (i + 1)))
            print('             <Gspy> = {:.3f}'.format(gspy / (i + 1)))
    GS.append(gs / numponderas)
    GSPY.append(gspy / numponderas)

file_name = "gainS_SPY__vs__spied_" + str(alpha) + 'alpha_' + str(
    M) + 'M_' + str(numponderas) + "ponds"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(GS)):
        file.write('{} {} {}\n'.format(Nspied[i], GS[i], GSPY[i]))

fig, ax = plt.subplots(2,
                       sharex=True,
                       gridspec_kw={'hspace': 0},
                       figsize=(10, 8))
ax[0].tick_params(which='both',
                  labelbottom=False,
                  labeltop=True,
                  labelleft=True,
                  labelright=False,
                  bottom=True,
                  top=True,
                  left=True,
                  right=True)
ax[1].tick_params(which='both',
                  labelbottom=True,
                  labeltop=False,
                  labelleft=True,
                  labelright=False,
                  bottom=True,
                  top=True,
                  left=True,
                  right=True)

ax[0].plot(Nspied, GSPY, marker='o', mec='navy', mfc='w', ls='-', color='k')
ax[0].set_xlabel('Mean gain Spy')
ax[0].axhline(y=0, color='k', linestyle='--')
ax[0].set_xscale('log')
ax[1].plot(Nspied, GS, marker='o', mec='navy', mfc='w', ls='-', color='k')
ax[1].set_xlabel('Mean gain other agents')
ax[1].set_xscale('log')
plt.show()
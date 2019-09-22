import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import GameSimulationInfo

emes = [(3 + i) for i in range(13)]
enes = [101, 121, 145]
alpha = []
for N in enes:
    for M in emes:
        alpha.append(2**M / N)

H = []
numponds = 10
file_name = 'HvsALPHA_' + '101,121,145N_' + str(numponds) + 'numponds'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[0].split('\n')
        H.append(float(value[0]))

alphab = []
Hbien = []
numponds = 100
file_name = 'HvsALPHA_' + '101,121,145N_' + str(numponds) + 'numponds'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alphab.append(float(value[0]))
        Hbien.append(float(value[1]))

H = [x for _, x in sorted(zip(alpha, H))]
alpha.sort()

alpha8 = []
theta8 = []
M = 8
file_name = "theta_vs_alpha_jupyter" + str(M) + "m"
with open('Data/INFO_vs_alpha/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha8.append(float(value[0]))
        theta8.append(float(value[1]))
del theta8[-1]
del theta8[-1]
del alpha8[-1]
del alpha8[-1]

#del H[15]
#del H[16]
#del alpha[15]
#del alpha[16]

fig, ax = plt.subplots()
ax.set(xscale='log', xlabel=r'$\alpha$', ylabel=r'$\theta^2$')
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

ax.plot(alpha8,
        theta8,
        color='k',
        alpha=0.8,
        marker='o',
        ms=5,
        mec='red',
        mfc='w',
        ls='')
ax.plot(alphab,
        np.array(Hbien) - 0.04,
        color='k',
        alpha=0.8,
        marker='o',
        ms=5,
        mec='green',
        mfc='w',
        ls='')
ax.axvline(x=0.337, color='k', alpha=0.6, linestyle='--')
#ax.set_yticks([0, 0.5, 1])
#ax.set_yticklabels(['0', '0.5', '1'])

plt.show()
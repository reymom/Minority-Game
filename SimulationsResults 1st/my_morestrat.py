import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import GameSimulationSPriv
from time import time

Ns = 101
Np = 51
M = 8

numponderas = 50
imprimecada = 1
used_strategies = []
gain_priv = []
esesprimas = []
HP = []
SP = 4
rango_s = 5  #31
for eses in range(rango_s):
    if eses < 10:
        SP += 1
    else:
        SP += 2
    esesprimas.append(SP)
    print("        ----------------------")
    print("        S' = {}        ({}/{})".format(SP, eses + 1, rango_s))
    print("        ----------------------")
    g = 0
    us = 0
    hp = 0
    t0 = time()
    for i in range(numponderas):
        gain, numstrats, H = GameSimulationSPriv(Ns, Np, M, SP, T=10000)
        g += np.mean(gain)
        us += numstrats
        hp += H
        if (i + 1) % imprimecada == 0:
            print('      ---> Ponderacion nº {}/{}'.format(i + 1, numponderas))
            print("      S' = {}        ({}/{})".format(SP, eses + 1, rango_s))
            print('            <gain> = {}'.format(g / (i + 1)))
            print('          <s_used> = {}'.format(us / (i + 1)))
            print('          <H/P> = {}'.format(hp / 2**M / (i + 1)))
            print('      tarda tiempo = ', (time() - t0) / (i + 1))
            print(' ')
    """gain_priv.append(g / numponderas)
    used_strategies.append(us / numponderas)"""
"""file_name = "PriviS_Sused_gain_" + str(Ns) + "Nspecs_" + str(
    Np) + 'Nprods_' + str(M) + 'm_' + str(numponderas) + "numponderas"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(esesprimas)):
        file.write('{} {} {}\n'.format(esesprimas[i], used_strategies[i],
                                       gain_priv[i]))"""
"""fig, ax = plt.subplots()
ax.set(xscale='log', xlabel="S'", ylabel="Nº strategies used")
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

ax.plot(esesprimas,
        used_strategies,
        color='k',
        alpha=0.6,
        marker='o',
        ms=6,
        mec='red',
        mfc='red',
        ls='')
plt.show()

fig, ax = plt.subplots()
ax.set(xscale='log', xlabel="S'", ylabel="Mean gain")
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

ax.plot(esesprimas,
        gain_priv,
        color='k',
        alpha=0.6,
        marker='s',
        ms=6,
        mec='blue',
        mfc='blue',
        ls='')
ax.axhline(y=0, color='k', alpha=0.7, linestyle='--')

plt.show()"""
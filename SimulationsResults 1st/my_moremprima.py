import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import one_simulation_morememo

alpha = 0.1
M = 3
N = int(2**M / alpha)  #da 80 justo

HMP = []
DIF = []
mis = 7
numponderas = 100
imprimecada = 20
for mi in range(mis):
    mp = M + mi
    hmp = 0
    gms = 0
    gmp = 0
    dif = 0
    print('------ mp = {} -------'.format(mp))
    for i in range(numponderas):
        t, H, HM, gain_s, gain_m = one_simulation_morememo(N, M, mp, T=15000)
        hmp += HM
        gms += gain_s
        gmp += gain_m
        dif += (gain_m - gain_s)
        if (i + 1) % imprimecada == 0:
            print('     Ponderacion nº {}/{}'.format(i + 1, numponderas))
            print('         Average_gs = {:.3f}'.format(gms / (i + 1)))
            print('         Average_gm = {:.3f}'.format(gmp / (i + 1)))
            print('         avera gm-gs = {:.3f}'.format(dif / (i + 1)))
            print('         Average_Hm/N = {:.3f}'.format(hmp / (i + 1)))
    HMP.append(hmp / numponderas)
    DIF.append(dif / numponderas)
    print(' ')

file_name = "deltaM__infos_difgains_" + str(numponderas) + "ponderas_cortas"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(mis):
        file.write('{} {} {}\n'.format(i, HMP[i], DIF[i]))

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
ax[0].plot(range(mis), HMP, marker='o', mec='b', mfc='w', ls='-')
ax[1].plot(range(mis), DIF, marker='o', mec='b', mfc='w', ls='-')
plt.show()
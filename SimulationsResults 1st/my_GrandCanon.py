import numpy as np
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import GameSimulationGC
import matplotlib.pyplot as plt

c = 0.5
M = 5

Ns = 107

PNP = np.linspace(1, 40, 20)
Nprods = PNP * 2**M
Nprods = Nprods.astype(int)
index_odd = np.argwhere(Nprods % 2 != 0).flatten()
Nprods[index_odd] = Nprods[index_odd] + 1
N_NP = Nprods / 2**M

print('Nprods[P]:\n', N_NP)
print('Nprods:\n', Nprods)

GS = []
GP = []
NPS = []
T = 50000
numponderas = 500
imprimecada = 1
for Np in [360]:  #Nprods:
    ys = 0
    yp = 0
    nps = 0
    for i in range(numponderas):
        meangain_s, meangain_p, nplayers_s = GameSimulationGC(Ns, Np, M, T)
        ys += np.mean(meangain_s)
        yp += np.mean(meangain_p)
        nps += np.mean(nplayers_s)
        if (i + 1) % imprimecada == 0:
            print('Ponderacion nº {}/{}'.format(i + 1, numponderas))
            print(' Nprods = {}        ({}/{})'.format(
                Np,
                list(Nprods).index(Np) + 1, len(Nprods)))
            print('    <gain_s> = {:.3f}'.format(ys / (i + 1)))
            print('    <gain_p> = {:.4f}'.format(yp / (i + 1)))
            print('    <Nplayers_s> = {:.2f}'.format(nps / (i + 1)))
            print(' ')
    GS.append(ys / numponderas)
    GP.append(yp / numponderas)
    NPS.append(nps / numponderas)
"""file_name = 'GainSpecProd_NPLAYERSvsPRODS_' + '107Ns_5M_' + str(
    numponderas) + 'numponds'
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(GS)):
        file.write('{} {} {} {}\n'.format(N_NP[i], GS[i], GP[i], NPS[i]))"""

fig, ax = plt.subplots()
ax.set(xlabel=r'$N_{prods}/2^M$', ylabel=r'$\left<G_i \right>$')
ax.tick_params(which='both',
               labelbottom=True,
               labeltop=False,
               labelleft=True,
               labelright=False,
               bottom=True,
               top=True,
               left=True,
               right=True)
ax.plot(N_NP,
        GS,
        marker='o',
        mec='blue',
        mfc='blue',
        ls='-',
        lw=0.7,
        alpha=0.5,
        c='k',
        label=r'$\left<G_{\{speculators\}}\right>$')

ax.plot(N_NP,
        GP,
        marker='s',
        mec='red',
        mfc='red',
        ls='-',
        lw=0.7,
        alpha=0.5,
        c='k',
        label=r'$\left<G_{\{producers\}}\right>$')

ax.axhline(y=-0.01, color='k', alpha=0.4, linestyle='--')

ax.legend(loc=0)
plt.show()

fig, ax = plt.subplots()
ax.tick_params(which='both',
               labelbottom=True,
               labeltop=False,
               labelleft=True,
               labelright=False,
               bottom=True,
               top=True,
               left=True,
               right=True)
ax.set(xlabel=r'$N_{prods}/2^M$', ylabel=r'$ \left<N_played> \right>$')
ax.plot(N_NP, NPS, marker='o', color='k', mec='red', mfc='red', alpha=0.8)
plt.show()
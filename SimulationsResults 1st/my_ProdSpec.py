import numpy as np
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import GameSimulationProducers
import matplotlib.pyplot as plt

c = 0
M = 8

T = 10000
numponderas = 100
imprimecada = 10

Np = 1024

SNP = np.linspace(1, 5, 16)
Nspecs = SNP * 2**M
Nspecs = Nspecs.astype(int)
index_even = np.argwhere(Nspecs % 2 == 0).flatten()
Nspecs[index_even] = Nspecs[index_even] + 1
S_NP = Nspecs / 2**M

print(Nspecs)

GS = []
GP = []
#0.5
for Ns in Nspecs:
    ys = 0
    yp = 0
    for i in range(numponderas):
        meangain_s, meangain_p = GameSimulationProducers(Ns, Np, M, T, c)
        ys += np.mean(meangain_s)
        yp += np.mean(meangain_p)
        if (i + 1) % imprimecada == 0:
            print('Ponderacion nº {}/{}'.format(i + 1, numponderas))
            print(' Nspecs = {}        ({}/{})'.format(
                Ns,
                list(Nspecs).index(Ns) + 1, len(Nspecs)))
            print('    <gain_s> = {}'.format(ys / (i + 1)))
            print('    <gain_p> = {}'.format(yp / (i + 1)))
            print(' ')
    GS.append(ys / numponderas)
    GP.append(yp / numponderas)

file_name = 'GainSpecProdvsProds_' + '1024Np_8M_' + str(
    numponderas) + 'numponds'
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(GS)):
        file.write('{} {} {}\n'.format(S_NP[i], GS[i], GP[i]))

file_name = 'GainSpecProdvsProds_' + '1024Np_8M_' + str(
    numponderas) + 'numponds'
with open('{}.dat'.format(file_name), 'w') as file:
    for i in range(len(GS)):
        file.write('{} {} {}\n'.format(S_NP[i], GS[i], GP[i]))

fig, ax = plt.subplots()
ax.set(xlabel=r'$N_{prods}/2^M$',
       ylabel=r'$\left<G_i \right>$',
       xlim=[-0.2, 9.2])
ax.tick_params(which='both',
               labelbottom=True,
               labeltop=False,
               labelleft=True,
               labelright=False,
               bottom=True,
               top=True,
               left=True,
               right=True)
ax.plot(S_NP,
        GS,
        marker='o',
        mec='blue',
        mfc='blue',
        ls='-',
        lw=0.7,
        alpha=0.5,
        c='k',
        label=r'$\left<G_{\{speculators\}}\right>$')

ax.plot(S_NP,
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
#ax.axvline(x=1.87, color='k', alpha=0.4, linestyle='--')
#ax.axvspan(-0.2, 1.87, color='y', alpha=0.1, lw=0)
#ax.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5])
#ax.set_yticklabels(['-2', None, '-1', None, '0', None])

ax.legend(loc=4)
plt.show()
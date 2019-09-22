import numpy as np
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
import matplotlib.pyplot as plt

Np = []
GS = []
GP = []
numponderas = 100
file_name = 'GainSpecProdvsProds_' + '641Ns_' + str(numponderas) + 'numponds'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[2].split('\n')
        Np.append(float(value[0]))
        GS.append(float(value[1]))
        GP.append(float(value[2]))
M = 8
Np = np.array(Np) / 2**M

Ns = []
GS2 = []
GP2 = []
numponderas = 100
file_name = 'GainSpecProdvsProds_' + '1024Np_8M_' + str(
    numponderas) + 'numponds'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[2].split('\n')
        Ns.append(float(value[0]))
        GS2.append(float(value[1]))
        GP2.append(float(value[2]))

fig, ax = plt.subplots(1, 2, gridspec_kw={'wspace': 0.05}, figsize=(12, 8))

ax[0].set(xlim=[-0.5, 9.5],
          ylim=[-2.05, 0.6])
ax[0].set_xlabel(r'$N_{prods}/2^M$', fontsize = 'large')
ax[0].set_ylabel(r'$<G_i>$', fontsize ='large')
ax[0].tick_params(which='both', length=3.5, width = 1.4, direction = 'inout',
                  labelbottom=True,
                  labeltop=False,
                  labelleft=True,
                  labelright=False,
                  bottom=True,
                  top=True,
                  left=True,
                  right=True)
ax[0].plot(Np,
           GS,
           marker='o',
           mec='blue',
           mfc='blue',
           ls='-',
           lw=0.7,
           alpha=0.5,
           c='k',
           label=r'$<G_{\{speculators\}}>$')

ax[0].plot(Np,
           GP,
           marker='s',
           mec='red',
           mfc='red',
           ls='-',
           lw=0.7,
           alpha=0.5,
           c='k',
           label=r'$<G_{\{producers\}}>$')

ax[0].axhline(y=-0.01, color='k', alpha=0.4, linestyle='--')
ax[0].axvline(x=1.87, color='k', alpha=0.4, linestyle='--')
ax[0].axvspan(-0.5, 1.87, color='y', alpha=0.1, lw=0)
ax[0].set_yticks([-2, -1.5, -1, -0.5, 0, 0.5])
ax[0].set_yticklabels(['-2', None, '-1', None, '0', None])

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.45, 0.15),framealpha=1)

#ax[0].legend(bbox_to_anchor=(0.9, 0.2))

ax[1].set(ylim=[-2.05, 0.6], xlim=[0.7, 5.3])
ax[1].set_xlabel(r'$N_{specs}/2^M$', fontsize='large')
ax[1].tick_params(which='both', length=3.5, width = 1.4, direction = 'inout',
                  labelbottom=True,
                  labeltop=False,
                  labelleft=False,
                  labelright=False,
                  bottom=True,
                  top=True,
                  left=True,
                  right=True)
ax[1].plot(Ns[:-2],
           GS2[:-2],
           marker='o',
           mec='blue',
           mfc='blue',
           ls='-',
           lw=0.7,
           alpha=0.5,
           c='k')
#,label=r'$\left<G_{\{speculators\}}\right>$')

ax[1].plot(Ns,
           GP2,
           marker='s',
           mec='red',
           mfc='red',
           ls='-',
           lw=0.7,
           alpha=0.5,
           c='k')
#,label=r'$\left<G_{\{producers\}}\right>$')

ax[1].axhline(y=-0.01, color='k', alpha=0.4, linestyle='--')
ax[1].axvline(x=3.3, color='k', alpha=0.4, linestyle='--')
ax[1].axvspan(3.3, 5.3, color='y', alpha=0.1, lw=0)
ax[1].set_xticks([1, 2, 3, 4, 5])
ax[1].set_xticklabels(['1', '2', '3', '4', '5'])
ax[1].set_yticks([-2, -1.5, -1, -0.5, 0, 0.5])
ax[1].set_yticklabels(['-2', None, '-1', None, '0', None])
#ax[1].legend(loc=3)

filename = 'GainSpecProd_vs_Nprods_0c_8M__641Ns__1024Np.png'
fig.savefig(filename, bbox_inches='tight')

#plt.show()
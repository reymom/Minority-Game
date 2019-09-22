import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")

alpha = []
G_S = []
G_M = []
HN_S = []
HN_M = []

numponderas = 100
file_name = "alpha__gains_infos_mprima_m_emes358" + str(
    numponderas) + "numponderas"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[4].split('\n')
        alpha.append(float(value[0]))
        G_S.append(float(value[1]))
        G_M.append(float(value[2]))
        HN_S.append(float(value[3]))
        HN_M.append(float(value[4]))

del G_M[8]
alphacopy = alpha.copy()
del alphacopy[8]
"""
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('H/N')
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
ax.plot(
    alpha[4:],
    HN_S[4:],
    marker='o',
    mec='blue',
    mfc='blue',
    #ls='-',
    #lw = 0.5,
    alpha=0.5,
    c='k',
    label='Players with M')
ax.plot(
    alpha[4:],
    HN_M[4:],
    marker='s',
    mec='red',
    mfc='red',
    #ls='-',
    #lw = 0.5,
    alpha=0.5,
    c='k',
    label="Player with M+1")
ax.legend(loc=0)
plt.show()
"""

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('average gain per agent')
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
               right=False)
ax.plot(alpha,
        G_S,
        marker='o',
        mec='k',
        mfc='w',
        ls='-',
        label='Players with M')
ax.plot(alphacopy,
        G_M,
        marker='o',
        mec='k',
        mfc='w',
        ls='-',
        label="Player with M+1")
ax.axhline(y=0, color='k', linestyle='--')
ax.legend(loc=0)

axin = fig.add_axes([0.3, 0.15, 0.6, 0.3])
axin.set_xscale('log')
#axin.set_xlabel(r'$\alpha$')
#axin.set_ylabel('average gain per agent')
axin.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))
axin.tick_params(which='both',
                 direction='in',
                 labelbottom=False,
                 labeltop=True,
                 labelleft=True,
                 labelright=False,
                 bottom=True,
                 top=True,
                 left=True,
                 right=True)
axin.plot(alpha[9:], G_S[9:], marker='o', mec='k', mfc='w', ls='-')
axin.plot(alphacopy[4:], G_M[4:], marker='o', mec='k', mfc='w', ls='-')
axin.axhline(y=0, color='k', linestyle='--')
#axin.set_xlim([3,1])
axin.tick_params(axis="x", labelsize='x-small')
axin.set_ylim([-1.8, 1])
axin.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1])
axin.set_yticklabels(['-1.5', '-1', '-0.5', '0', '0.5', None],
                     fontsize='x-small')

plt.show()
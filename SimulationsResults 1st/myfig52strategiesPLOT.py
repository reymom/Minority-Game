import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")


alpha2 = []
sigma2 = []
S2 = 2
numponderas = 25
file_name = 'VARvsALPHA_' + str(S2) + 'S_' + '101,143' + 'N_' + str(
    numponderas) + 'num_ponderas'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha2.append(float(value[0]))
        sigma2.append(float(value[1]))


alpha3 = []
sigma3 = []
S3 = 3
numponderas = 25
file_name = 'VARvsALPHA_' + str(S3) + 'S_' + '101,143' + 'N_' + str(
    numponderas) + 'num_ponderas'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha3.append(float(value[0]))
        sigma3.append(float(value[1]))

alpha4 = []
sigma4 = []
S4 = 4
numponderas = 25
file_name = 'VARvsALPHA_' + str(S4) + 'S_' + '101,143' + 'N_' + str(
    numponderas) + 'num_ponderas'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha4.append(float(value[0]))
        sigma4.append(float(value[1]))

alpha6 = []
sigma6 = []
S6 = 6
numponderas = 25
file_name = 'VARvsALPHA_' + str(S6) + 'S_' + '101,143' + 'N_' + str(
    numponderas) + 'num_ponderas'
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha6.append(float(value[0]))
        sigma6.append(float(value[1]))


fig, ax = plt.subplots()
ax.set(xscale = 'log', yscale = 'log', xlabel = r'$\alpha$', ylabel = r'$\sigma^2/N$')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.tick_params(which='both', 
               labelbottom=True, labeltop=False,labelleft=True, labelright=False,
               bottom=True, top=True, left=True, right=True)
ax.plot(alpha2, sigma2, color = 'k', alpha = 0.8, marker= 'o', ms = 5,
                    mec = 'indianred', mfc= 'w', ls='-', label=r'$S = 2$')
ax.plot(alpha3, sigma3, color = 'k', alpha = 0.8, marker= '^', ms = 5,
                    mec = 'forestgreen', mfc= 'w', ls='-', label=r'$S = 3$')
ax.plot(alpha4, sigma4, color = 'k', alpha = 0.8, marker= 's', ms = 5,
                    mec = 'navy', mfc= 'w', ls='-', label=r'$S = 4$')
ax.plot(alpha6, sigma6, color = 'k', alpha = 0.8, marker= '*', ms = 5,
                    mec = 'darkgoldenrod', mfc= 'w', ls='-', label=r'$S = 6$')

ax.axhline(y=1, xmin = 0, xmax = 1, color='k', alpha = 0.8, linestyle='--')
ax.legend(loc=0)
plt.show()
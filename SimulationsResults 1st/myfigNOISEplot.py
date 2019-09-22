import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")

num_ponderas = 100
alpha_noise = []
var_noise = []
file_name = "var_vs_alpha_noise_" + "variableN_" + 'variableNn_' + str(
    num_ponderas) + "num_ponderas"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha_noise.append(float(value[0]))
        var_noise.append(float(value[1]))

num_ponderas = 100
alpha_normal = []
var_normal = []
file_name = "var_H_vs_alpha_normal_" + "variableN_" + 'variableNn_' + str(
    num_ponderas) + "num_ponderas"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha_normal.append(float(value[0]))
        var_normal.append(float(value[1]))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma^2/N$')

ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.set_ylim(0, 3)
ax.plot(alpha_noise[4:],
        var_noise[4:],
        marker='o',
        ms=5,
        mec='k',
        mfc='w',
        ls='--',
        c='red',
        label=r'$N = N_{normal} + N_{noise}$')
ax.plot(alpha_normal[1:],
        var_normal[1:],
        marker='s',
        ms=5,
        mec='k',
        mfc='w',
        ls='--',
        c='blue',
        label=r'$N = N_{normal}$')
ax.axhline(y=1, color='k', linestyle='--')
ax.legend(loc=0)
plt.show()
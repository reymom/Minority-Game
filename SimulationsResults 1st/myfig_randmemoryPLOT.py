import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")

S = 2
N = 101
num_ponderas = 100

emes = []
var = []
err = []
file_name = "var_vs_alpha_MEMORY_errors_"+str(N)+"N_"+str(num_ponderas)+"num_ponderas"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[2].split('\n')
        emes.append(float(value[0]))
        var.append(float(value[1]))
        err.append(float(value[2]))

emes0 = []
var0 = []
err0 = []
file_name = "var_vs_alpha_MEMORYFAKE_errors_"+str(N)+"N_"+str(num_ponderas)+"num_ponderas"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[2].split('\n')
        emes0.append(float(value[0]))
        var0.append(float(value[1]))
        err0.append(float(value[2]))

fig, ax = plt.subplots()
ax.set_xlabel('$m$')
ax.set_ylabel(r'$\sigma$')
ax.set_xlim(0,15)
ax.errorbar(emes, np.sqrt(var), yerr=np.sqrt(err)/2, capsize = 5, color='k', 
                            marker= '.', mec = 'navy', mfc= 'w', ls='-', 
                            lw=0.5, label = 'With memory record')
ax.errorbar(emes0, np.sqrt(var0), yerr=np.sqrt(err0)/2, capsize = 5, 
                            marker= '.', mec = 'navy', mfc= 'w', color='red', 
                            ls='', label = 'With random memory')
ax.axhline(y=10, xmin= 1/15, xmax=14/15, color='k', linestyle='--')
ax.legend()
plt.show()
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import one_simulation_variables, one_simulation_fake_memory

S = 2
N = 101

emes = [(2+i) for i in range(12)]
var = []  # Y
errors = []

num_ponderas = 100
for M in emes:
    print(' M = {}               ({}/{})'.format(M, emes.index(M)+1, len(emes)))
    y = []
    orinan = []
    for pond in range(num_ponderas):
        print('     Ponderation nº{} of {}'.format(pond+1, num_ponderas))
        times, attendances = one_simulation_fake_memory(N, S, M, T=10000, imprime=5000)
        y.append(np.var(attendances))
        orinan.append(np.mean(attendances))
    var.append(np.mean(y))
    errors.append(stats.sem(y))
    print('σ²/N = {} ± {}'.format(var[-1]/N,errors[-1]/N))
    print('<A> = {}'.format(np.mean(orinan)))
    print(' ')

#np.save("var601.npy", var)

file_name = "var_vs_alpha_MEMORYFAKE_errors_"+str(N)+"N_"+str(num_ponderas)+"num_ponderas"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(var)):
        file.write('{} {} {}\n'.format(emes[i], var[i], errors[i]))

fig, ax = plt.subplots()
ax.set_xlabel('$m$')
ax.set_ylabel(r'$\sigma$')
ax.set_xlim(0,15)
ax.errorbar(emes, np.sqrt(var), yerr=np.sqrt(errors)/2, capsize = 5, color='k', 
                            marker= 'o', mec = 'navy', mfc= 'w', ls='-')
ax.axhline(y=10, xmin= 1/15, xmax=14/15, color='k', linestyle='--')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import one_simulation_fake_memory, one_simulation_noise

S = 2
emes = [(3 + i) for i in range(10)]

H = []
alpha = []  # X
var = []  # Y
num_ponderas = 100
i = -1
for M in emes:
    i += 1
    j = -1
    for N, Nnoise in zip([131, 85], [65, 42]):
        j += 1
        alf = 2**M / (N + Nnoise)
        alpha.append(alf)
        print('   SIMULATION NUMBER {}/{}'.format(
            emes.index(M) * 2 + j + 1,
            len(emes) * 2))
        print('   N = {} ------- Nnoise = {}'.format(N, Nnoise))
        print('   alpha = {:.3f}'.format(alf))
        y = []
        orinan = []
        for pond in range(num_ponderas):
            print('     Ponderation nº{} of {}'.format(pond + 1, num_ponderas))
            times, attendances = one_simulation_noise(N, Nnoise, S, M, T=10000)
            #times, attendances = one_simulation_fake_memory(N, S, M, T=10000)
            y.append(np.var(attendances))
            orinan.append(np.mean(attendances))
        var.append(np.mean(y) / (N + Nnoise))
        H.append(np.mean(y)**2)
        print('σ²/N = ', var[-1])
        print('H = ', H[-1])
        print(' ')

file_name = "var_H_vs_alpha_noise_" + "variableN_" + 'variableNn_' + str(
    num_ponderas) + "num_ponderas"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(var)):
        file.write('{} {} {}\n'.format(alpha[i], var[i], H[i]))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma$')
#ax.yaxis.set_major_formatter(
#    ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(
#        int(np.maximum(-np.log10(y), 0)))).format(y)))
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.plot(alpha, np.array(var), marker='o', mec='k', mfc='w', ls='-')
ax.axhline(y=1, color='k', linestyle='--')
plt.show()
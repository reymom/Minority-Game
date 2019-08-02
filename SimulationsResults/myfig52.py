from mygame_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

S = 2
N = 501

emes = [(2+i) for i in range(12)]
alpha = [2**i/N for i in emes]  # X
var = []  # Y

num_ponderas = 50
for M in emes:
    print(' M = {}               ({}/{})'.format(M, emes.index(M)+1, len(emes)))
    print(' α = {:.3f}'.format(alpha[emes.index(M)]))
    y = []
    orinan = []
    for pond in range(num_ponderas):
        # if pond % num_ponderas/10 == 0:
        print('     Ponderation nº{} of {}'.format(pond+1, num_ponderas))
        times, attendances = one_simulation(N, S, M, T=7500, imprime=2500)
        y.append(np.var(attendances))
        orinan.append(np.mean(attendances))
        #print('     var/N = {:.2f}'.format(y[-1]/N))
        #print('     mean = {:4f}'.format(orinan[-1]))
    var.append(np.mean(y))
    print('σ²/N = ', var[-1]/N)
    print('<A> = ', np.mean(orinan))
    print(' ')

np.save("var501.npy", var)

file_name = "var_vs_alpha_"+str(N)+"N_"+str(num_ponderas)+"num_ponderas"
with open('Minority_Game/Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(var)):
        file.write('{} {}\n'.format(alpha[i], var[i]))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma$')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.plot(alpha, np.array(var)/N, '.')
plt.show()

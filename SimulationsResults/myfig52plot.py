from mygame_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

alpha101 = []
var101 = []
S = 2
N101 = 101
num_ponderas = 50
file_name = "var_vs_alpha_"+str(N101)+"N_"+str(num_ponderas)+"num_ponderas"
with open('Minority_Game/Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha101.append(float(value[0]))
        var101.append(float(value[1]))

alpha301 = []
var301 = []
S = 2
N301 = 301
num_ponderas = 40
file_name = "var_vs_alpha_"+str(N301)+"N_"+str(num_ponderas)+"num_ponderas"
with open('Minority_Game/Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha301.append(float(value[0]))
        var301.append(float(value[1]))

alpha501 = []
var501 = []
S = 2
N501 = 501
num_ponderas = 50
file_name = "var_vs_alpha_"+str(N501)+"N_"+str(num_ponderas)+"num_ponderas"
with open('Minority_Game/Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha501.append(float(value[0]))
        var501.append(float(value[1]))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma^2/N$')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.plot(alpha101, np.array(var101)/N101, '.', label=r'$N = 101$')
ax.plot(alpha301, np.array(var301)/N301, '.', label=r'$N = 301$')
ax.plot(alpha501, np.array(var501)/N501, '.', label=r'$N = 501$')
ax.axhline(y=1, color='k', linestyle='--')
ax.legend(loc=0)
plt.show()

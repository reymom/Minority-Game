from mygame_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

alpha = []
theta2 = []
S = 2
M = 8
file_name = "theta_vs_alpha_"+str(M)+"m"
with open('Minority_Game/Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha.append(float(value[0]))
        theta2.append(float(value[1]))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\theta^2$')
ax.set_title('m=8')
ax.plot(alpha, theta2, '.')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
plt.show()

from mygame_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


S = 2
M = 8
alpha = [10**(-0.6 + 0.0815*ni)  # 31
         for ni in range(21)]  # Logaritmically equispaced
enes = [int(2**M/alpha[i]) for i in range(len(alpha))]  # N has to be integer
alpha = [2**M/enes[i] for i in range(len(alpha))]  # X
T = [500, 600, 700, 800, 900, 1000, 1000, 1000, 1000, 1000, 1000,
     3000, 5000, 6000, 8000, 9000, 100000, 10000, 10000, 10000, 10000]
num_ponderas = 50
theta_2 = []
for N in enes:
    print('N={} ({}/{})'.format(N, enes.index(N)+1, len(enes)))
    print(r'alpha = {:.3f}'.format(alpha[enes.index(N)]))
    y = []
    for j in range(num_ponderas):
        print('     Ponderacion {}/{}'.format(j+1, num_ponderas))
        times, attendances, information = one_simulation_info(
            N, S, M, T[i], imprime=250)
        y.append(np.mean(information))
    # print(information)
    theta_2.append(np.mean(y))
    print(theta_2[-1])

file_name = "theta_vs_alpha_conponderas"+str(M)+"m"
with open('Minority_Game/Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(theta_2)):
        file.write('{} {}\n'.format(alpha[i], theta_2[i]))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('H')
ax.plot(alpha, theta_2, '.')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: (
    '{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
plt.show()

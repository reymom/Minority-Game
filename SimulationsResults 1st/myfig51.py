from mygame_functions import *
import numpy as np
import matplotlib.pyplot as plt

T = 50
N = 9
S = 2
M = 2
times, attendances = one_simulation(N, S, M, T)

fig = plt.figure(1, figsize=(8, 4))
plt.plot(times, attendances, linewidth=0.8)
#plt.ylim(-500, 500)
plt.xlabel(r'$t$')
plt.ylabel(r'$A(t)/N$')
fig.set_tight_layout(True)
plt.show()

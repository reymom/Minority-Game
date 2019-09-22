import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")
from mygame_functions import one_simulation_grandcanon

S = 2
c = 0.5

Nspecs = 13
M = 2
P = 2**M

T = 10
num_ponderas = 1

NUMP = []
GS = []
GP = []
PLAY_S = []
AS = []
for n in range(2):
    nP = 2.5 * (1 + n)
    Nprods = int(nP * P)
    NUMP.append(nP)
    print('nP = {}, Nprods = {}, sim {}/{}'.format(nP, Nprods, n + 1, 24))
    ys = 0
    yp = 0
    players = 0
    atts = 0
    for i in range(num_ponderas):
        print('Ponderacion nº {}/{}'.format(i + 1, num_ponderas))
        t, A, mean_gain_specs, mean_gain_prods, s_played_t = one_simulation_grandcanon(
            Nspecs, Nprods, S, M, T)
        ys_t = np.mean(mean_gain_specs)
        ys += ys_t
        yp_t = np.mean(mean_gain_prods)
        yp += yp_t
        players_t = np.mean(s_played_t)
        players += players_t
        atts += np.mean(A)
        print('   -<A> = ', -atts / (i + 1))
        print('   <s_gain> = ', ys / (i + 1))
        print('   <p_gain> = ', yp / (i + 1))
        print('   <numplayers> = ', players / (i + 1))
    AS.append(-atts / num_ponderas)
    GS.append(ys / num_ponderas)
    GP.append(yp / num_ponderas)
    PLAY_S.append(players / num_ponderas)

    print(' ')
    #print('<s_gain> = {}'.format(GS[-1]))
    #print('<p_gain> = {}'.format(GP[-1]))
    #print('<s_players> = {}'.format(PLAY_S[-1]))
    #print(' ')
    print(' ')

file_name = "Nprod_Pgain_Sgain_Splayers_" + str(Nspecs) + "Nspecs_" + str(
    c) + 'c_' + str(M) + 'm_' + str(num_ponderas) + "num_ponderas"
with open('Data/{}.dat'.format(file_name), 'w') as file:
    for i in range(len(GS)):
        file.write('{} {} {} {}\n'.format(NUMP[i], AS[i], GP[i], GS[i],
                                          PLAY_S[i]))

fig, ax = plt.subplots()
ax.plot(NUMP, GS, label=r'$G_{spec}$')
ax.plot(NUMP, GP, label=r'$G_{prod}$')
ax.set_xlabel(r'$Nº$ of producers [P]')
ax.set_ylabel('Average gain per agent')
ax.legend(loc=0)
plt.show()

fig, ax = plt.subplots()
ax.plot(NUMP, PLAY_S)
ax.set_xlabel(r'$Nº$ of producers [P]')
ax.set_ylabel('Average number of speculators in the market')
plt.show()
import numpy as np
import networkx as nx
import random

#----------------------------------------------------#
#----------------------NETWORKS----------------------#
#----------------------------------------------------#


def Neighbors_Lattice(N):
    """
    -----------------
    Input
            Number of nodes N, where N = L*L, L being the side of the square lattice
    -----------------
    Output
            An array of N elements
                groups of 4 elements are the neighbors of the ith node
                according to the structure of a square lattice (2D)
    -----------------
    In order to take the neighbors for each node individually do that
    for i in range(N):
        print(neighs[4*i:(4*(i+1))])
    -----------------
    """
    L = int(np.sqrt(N))
    sites = np.arange(L * L).reshape(L, L)
    neighs = []
    for i in range(L):
        for j in range(L):
            for k in [-1, 1]:
                neighs.append(sites[(i + k) % L, j])
                neighs.append(sites[i, (j + k) % L])
    return np.array(neighs)


def Neighbors_SmallWorld(N, k, p):
    """
    -----------------
    Inputs
            N: the number of nodes
            k: the number of neighbours of each node
            p: the probability of rewiring
    -----------------
    Output 
             The neighbors according to the structure of the Watts-Strogatz model
                    ---> in an array of N elements
             An array of the position of the first neighbor to last for each node
    -----------------
    In order to get the neighbors for each node in an individual array, use this:
    
    for i in range(len(counts)-1):
        neighs[counts[i]:counts[i+1]]
    -----------------
    """
    G = nx.watts_strogatz_graph(N, k, p)
    to_neighs = [[] for i in range(N)]
    for i in G.edges:
        to_neighs[i[0]].append(i[1])
        to_neighs[i[1]].append(i[0])

    neighs = []
    counts = [0]
    cum = 0
    for i in to_neighs:
        cum += len(i)
        counts.append(cum)
        for j in i:
            neighs.append(j)
    return np.array(counts), np.array(neighs)


def Neighbors_R(N, grado):
    k = N*grado/2
    G = nx.gnm_random_graph(N, k)
    to_neighs = [[] for i in range(N)]
    for i in G.edges:
        to_neighs[i[0]].append(i[1])
        to_neighs[i[1]].append(i[0])
    neighs = []
    counts = [0]
    cum = 0
    for i in to_neighs:
        cum += len(i)
        counts.append(cum)
        for j in i:
            neighs.append(j)
    return np.array(counts), np.array(neighs)


def Neighbors_BA(N, m_ba):
    """
    In order to get the neighbors for each node in an individual array, use this:
    
    for i in range(len(counts)-1):
        neighs[counts[i]:counts[i+1]]
    """
    G = nx.barabasi_albert_graph(N, m_ba)
    to_neighs = [[] for i in range(N)]
    for i in G.edges:
        to_neighs[i[0]].append(i[1])
        to_neighs[i[1]].append(i[0])
    neighs = []
    counts = [0]
    cum = 0
    for i in to_neighs:
        cum += len(i)
        counts.append(cum)
        for j in i:
            neighs.append(j)
    return np.array(counts), np.array(neighs)


def Neighbors_SocialNetwork():
    G = nx.read_edgelist("facebook_combined.txt",
                         create_using=nx.Graph(),
                         nodetype=int)
    N = len(G.nodes)
    to_neighs = [[] for i in range(N)]
    for i in G.edges:
        to_neighs[i[0]].append(i[1])
        to_neighs[i[1]].append(i[0])
        to_neighs = []
    counts = [0]
    cum = 0
    for i in to_neighs:
        cum += len(i)
        counts.append(cum)
        for j in i:
            neighs.append(j)
    return np.array(neighs), np.array(counts)


def Reduced_Neighbors(counts, neighs, nodes_imitators):
    """
    Returns the array of neighbors and its positions for the nodes selected in nodes_copying
    """
    reduced_counts = [0]
    reduced_neighs = []
    cont = 0
    for i in nodes_imitators:
        for j in neighs[counts[i]:counts[i + 1]]:
            reduced_neighs.append(j)
        cont += (counts[i + 1] - counts[i])
        reduced_counts.append(cont)
    return np.array(reduced_counts), np.array(reduced_neighs)


#-----------------------------------------------------#
#----------------GAMES AND SIMULATIONS----------------#
#-----------------------------------------------------#

#----------------------------#
#----------LOCAL MG----------#
#----------------------------#

def RandomImitationFors(strategies, scores, nodes_imitators, counts, neighs):
    chosen_neighs = np.zeros(len(counts) - 1)
    for i in range(len(counts) - 1):
        scores_min_imitators = np.argmin(scores[nodes_imitators, :], axis=1)
        neigh_copy = np.random.choice(neighs[counts[i]:counts[i + 1]])
        chosen_neighs[i] = neigh_copy
    chosen_neighs = chosen_neighs.astype(int)

    scores_max = np.argmax(scores, axis=1)
    scores_min_imitators = np.argmin(scores[nodes_imitators, :], axis=1)
    strategies_past = strategies.copy()
    scores_past = scores.copy()
    for i in range(len(counts) - 1):
        node_imit = nodes_imitators[i]
        #neigh_chosen = np.random.choice(neighs[counts[i]:counts[i + 1]])
        max_node_imit = scores_max[node_imit]
        min_node_imit = scores_min_imitators[i]
        max_chosen_neigh = scores_max[chosen_neighs[i]]
        if scores_past[node_imit, max_node_imit] < scores_past[
                chosen_neighs[i], max_chosen_neigh]:
            strategies[node_imit, min_node_imit] = strategies_past[
                chosen_neighs[i], max_chosen_neigh]
            scores[node_imit, min_node_imit] = 0
    return strategies, scores, scores_max


def RandomImitationArrays(strategies, scores, nodes_imitators, counts, neighs):
    chosen_neighs = np.zeros(len(counts) - 1)
    for i in range(len(counts) - 1):
        node = nodes_imitators[i]
        neigh_copy = np.random.choice(neighs[counts[i]:counts[i + 1]])
        chosen_neighs[i] = neigh_copy
    chosen_neighs = chosen_neighs.astype(int)

    scores_max = np.argmax(scores, axis=1)
    scores_max_imitators = scores_max[nodes_imitators]
    scores_max_neighs = scores_max[chosen_neighs]
    scores_min_imitators = np.argmin(scores[nodes_imitators], axis=1)
    strategies_past = strategies.copy()
    copy_from = np.argwhere(
        10*scores[nodes_imitators, scores_max_imitators] <
        scores[chosen_neighs, scores_max_neighs]).flatten()
    strategies[nodes_imitators[copy_from],
               scores_min_imitators[copy_from], :] = strategies_past[
                   chosen_neighs[copy_from], scores_max_neighs[copy_from], :]
    scores[nodes_imitators[copy_from], scores_min_imitators[copy_from]] = 0
    
    #scores[chosen_neighs[copy_from], scores_max_neighs[copy_from]]
    return strategies, scores

"""def ImitationMaximum(strategies, scores, S, nodes_imitators, counts, neighs):
    scores_max = np.argmax(scores, axis=1)
    scores_min_imitators = np.argmin(scores[nodes_imitators], axis=1)
    maxneighs = np.zeros(len(nodes_imitators)).astype(int)
    numstratneigh = np.zeros(len(nodes_imitators)).astype(int)
    for i in range(len(counts) - 1):
        node = nodes_imitators[i]
        maximoneigh = np.amax(scores[neighs[counts[i]:counts[i + 1]]].flatten())
        iguales = np.argwhere(
            scores[neighs[counts[i]:counts[i + 1]]].flatten() == maximoneigh).flatten()
        cual = np.random.choice(iguales)
        strat = cual%S
        neigh_con_maximo = neighs[counts[i]:counts[i + 1]][int(cual/S)]
        maxneighs[i] = neigh_con_maximo
        numstratneigh[i] = strat
    copy_from = np.argwhere(scores[nodes_imitators, scores_max[nodes_imitators]] < scores[
        maxneighs, numstratneigh]).flatten()
    strategies[nodes_imitators[copy_from],scores_min_imitators[copy_from],:] = strategies[
                                        maxneighs[copy_from], numstratneigh[copy_from], :]
    scores[nodes_imitators[copy_from], scores_min_imitators[copy_from]] = 0
    return strategies, scores"""

def GameStepImitationSW(S, M, N, Ncop, t, dynT, counts, neighs, nodes, nodes_imitators,
                        strategies, scores):
    #DYNAMICS OF IMITATION
    if (t+1)%dynT == 0:
        """strategies, scores = RandomImitationArrays(strategies, scores,
                                                        nodes_imitators, counts,
                                                        neighs)"""
        strategies, scores = ImitationMaximum(strategies, scores, S,
                                                        nodes_imitators, counts,
                                                        neighs)    
    #DYNAMICS OF MINORITY GAME
    scores_max = np.argmax(scores, axis=1)
    scores_equal = np.argwhere((scores[:, 0] == scores[:, 1])).flatten()
    state = np.random.randint(2**M)
    ai = np.zeros(N)
    for nm in range(N):
        ai[nm] = strategies[nm, scores_max[nm], state]
    if len(scores_equal) > 0:
        for ne in scores_equal:
            ai[ne] = np.random.choice(strategies[ne, :, state])
    A = np.sum(ai)
    gains = -A * ai[nodes]
    gains_copy = -A * ai[nodes_imitators]
    scores = np.add(scores, -A * strategies[:, :, state])
    return A, scores, gains, gains_copy


def GameSimulationImitationSW(S, M, N, Ncop, p, T, dynT):
    strategies = 2 * np.random.randint(2, size=(N, S, 2**M)) - 1
    scores = np.zeros((N, S))
    counts, neighs = Neighbors_SmallWorld(N, p)
    nodes_imitators = np.array(random.sample(range(N), Ncop))
    nodes_imitators.sort()
    nodes = []
    for i in range(N):
        if i not in nodes_imitators:
            nodes.append(i)
    nodes = np.array(nodes)
    counts, neighs = Reduced_Neighbors(counts, neighs, nodes_imitators)
    A_t = np.zeros(T)
    G_t = np.zeros(T)
    GC_t = np.zeros(T)
    for t in range(T):
        A, scores, gains, gains_copy = GameStepImitationSW(S, M, N, Ncop, t, dynT,
                    counts, neighs, nodes, nodes_imitators, strategies, scores)
        A_t[t] = A
        G_t[t] = np.mean(gains)
        GC_t[t] = np.mean(gains_copy)
    return A_t, G_t, GC_t


def MaxImitationAll_proofs(strategies, scores, N, S, nodes_imitators):
    scores_max_imitators = np.argmax(scores[nodes_imitators], axis=1)
    scores_min_imitators = np.argmin(scores[nodes_imitators], axis=1)
    maxneighs = np.zeros(len(nodes_imitators)).astype(int)
    numstratneigh = np.zeros(len(nodes_imitators)).astype(int)
    for i in range(len(nodes_imitators)):
        node = nodes_imitators[i]
        maximoneigh = np.amax(scores.flatten())
        iguales = np.argwhere(
            scores.flatten() == maximoneigh).flatten()
        cual = np.random.choice(iguales)
        maxneighs[i] = int(cual/S)
        numstratneigh[i] = cual%S
    copy_from = np.argwhere(scores[nodes_imitators, scores_max_imitators] < scores[
        maxneighs, numstratneigh]).flatten()
    strategies[nodes_imitators[copy_from],scores_min_imitators[copy_from],:] = strategies[
                                        maxneighs[copy_from], numstratneigh[copy_from], :]
    scores[nodes_imitators[copy_from], scores_min_imitators[copy_from]] = 0
    return strategies, scores, maxneighs, numstratneigh


def GameStepImitationCG_proofs(S, M, N, Ncop, t, dynT, nodes, nodes_imitators,
                        strategies, scores):
    #DYNAMICS OF MINORITY GAME
    scores_max = np.argmax(scores, axis=1)
    scores_equal = np.argwhere((scores[:, 0] == scores[:, 1])).flatten()
    scores_max[scores_equal] = np.random.randint(S,size=len(scores_equal))
    
    state = np.random.randint(2**M)
    ai = strategies[np.arange(N),scores_max,state]
    
    A = np.sum(ai)
    gains = -A * ai
    scores = np.add(scores, -A * strategies[:, :, state])

    #DYNAMICS OF IMITATION
    ha_imitado = -100
    numstrat = -100
    if (t+1)%dynT == 0:
        strategies, scores, ha_imitado, numstrat = MaxImitationAll_proofs(
            strategies, scores, N, S, nodes_imitators)
    
    return A, scores, gains, ha_imitado, numstrat


def GameSimulationImitationCG_proofs(S, M, N, Ncop, T, dynT):
    strategies = 2 * np.random.randint(2, size=(N, S, 2**M)) - 1
    scores = np.zeros((N, S))

    nodes_imitators = np.array(random.sample(range(N), Ncop))
    nodes_imitators.sort()
    nodes = []
    for i in range(N):
        if i not in nodes_imitators:
            nodes.append(i)
    nodes = np.array(nodes)

    A_t = np.zeros(T)
    successrate=np.zeros((T,N))
    G_t = np.zeros(T)
    GI_t = np.zeros(T)
    SI_t = np.zeros((T,N,S))
    ha_imi = np.zeros(T)
    nu_str = np.zeros(T)
    for t in range(T):
        A, scores, gains, ha_imitado, numstrat = GameStepImitationCG_proofs(S, M, N, Ncop, 
                                        t, dynT, nodes, nodes_imitators, strategies, scores)
        A_t[t] = A
        G_t[t] = np.mean(gains[nodes])
        GI_t[t] = np.mean(gains[nodes_imitators])
        SI_t[t] = scores
        ha_imi[t] = ha_imitado
        nu_str[t] = numstrat
        success = gains/np.abs(gains)
        success[success<0] = 0
        successrate[t] = success.astype(int)
    return A_t, successrate, G_t, GI_t, SI_t, nodes_imitators, ha_imi, nu_str


######
def MaxImitationAll(strategies, scores, N, S, nodes_imitators):
    scores_max_imitators = np.argmax(scores[nodes_imitators], axis=1)
    scores_min_imitators = np.argmin(scores[nodes_imitators], axis=1)
    maxneighs = np.zeros(len(nodes_imitators)).astype(int)
    numstratneigh = np.zeros(len(nodes_imitators)).astype(int)
    for i in range(len(nodes_imitators)):
        node = nodes_imitators[i]
        maximoneigh = np.amax(scores.flatten())
        iguales = np.argwhere(
            scores.flatten() == maximoneigh).flatten()
        cual = np.random.choice(iguales)
        maxneighs[i] = int(cual/S)
        numstratneigh[i] = cual%S
    copy_from = np.argwhere(scores[nodes_imitators, scores_max_imitators] < scores[
        maxneighs, numstratneigh]).flatten()
    strategies[nodes_imitators[copy_from],scores_min_imitators[copy_from],:] = strategies[
                                        maxneighs[copy_from], numstratneigh[copy_from], :]
    scores[nodes_imitators[copy_from], scores_min_imitators[copy_from]] = 0
    return strategies, scores


def GameStepImitationCG(S, M, N, Ncop, t, dynT, nodes, nodes_imitators,
                        strategies, scores):

    #DYNAMICS OF IMITATION
    if (t+1)%dynT == 0:
        strategies, scores = MaxImitationAll(strategies, scores, N, S,
                                                        nodes_imitators)

    #DYNAMICS OF MINORITY GAME
    scores_max = np.argmax(scores, axis=1)
    scores_equal = np.argwhere((scores[:, 0] == scores[:, 1])).flatten()
    scores_max[scores_equal] = np.random.randint(S,size=len(scores_equal))
    
    state = np.random.randint(2**M)
    ai = strategies[np.arange(N),scores_max,state]
    
    A = np.sum(ai)
    gains = -A * ai
    scores = np.add(scores, -A * strategies[:, :, state])
    
    return A, scores, gains


def GameSimulationImitationCG(S, M, N, Ncop, T, dynT):
    strategies = 2 * np.random.randint(2, size=(N, S, 2**M)) - 1
    scores = np.zeros((N, S))

    nodes_imitators = np.array(random.sample(range(N), Ncop))
    nodes_imitators.sort()
    nodes = []
    for i in range(N):
        if i not in nodes_imitators:
            nodes.append(i)
    nodes = np.array(nodes)

    A_t = np.zeros(T)
    G_t = np.zeros(T)
    GI_t = np.zeros(T)

    for t in range(T):
        A, scores, gains = GameStepImitationCG(S, M, N, Ncop, t, dynT,
                                        nodes, nodes_imitators, strategies, scores)
        A_t[t] = A
        G_t[t] = np.mean(gains[nodes])
        GI_t[t] = np.mean(gains[nodes_imitators])

    return A_t, G_t, GI_t


##
def MaxImitationTermal(strategies, scores, N, S, imitators_now):
    scores_max_imitators = np.argmax(scores[imitators_now], axis=1)
    scores_min_imitators = np.argmin(scores[imitators_now], axis=1)
    maxneighs = np.zeros(len(imitators_now)).astype(int)
    numstratneigh = np.zeros(len(imitators_now)).astype(int)
    for i in range(len(imitators_now)):
        node = imitators_now[i]
        maximoneigh = np.amax(scores.flatten())
        iguales = np.argwhere(
            scores.flatten() == maximoneigh).flatten()
        cual = np.random.choice(iguales)
        maxneighs[i] = int(cual/S)
        numstratneigh[i] = cual%S
    copy_from = np.argwhere(scores[imitators_now, scores_max_imitators] < scores[
        maxneighs, numstratneigh]).flatten()
    strategies[imitators_now[copy_from],scores_min_imitators[copy_from],:] = strategies[
                                        maxneighs[copy_from], numstratneigh[copy_from], :]
    scores[imitators_now[copy_from], scores_min_imitators[copy_from]] = 0
    return strategies, scores


def GameStepImitationTermal(S, M, N, Ncop, t, nodes, imitators_now,
                        strategies, scores):

    #DYNAMICS OF IMITATION
    if len(imitators_now) > 0:
        strategies, scores = MaxImitationTermal(strategies, scores, N, S,
                                                        imitators_now)

    #DYNAMICS OF MINORITY GAME
    scores_max = np.argmax(scores, axis=1)
    scores_equal = np.argwhere((scores[:, 0] == scores[:, 1])).flatten()
    scores_max[scores_equal] = np.random.randint(S,size=len(scores_equal))
    
    state = np.random.randint(2**M)
    ai = strategies[np.arange(N),scores_max,state]
    
    A = np.sum(ai)
    gains = -A * ai
    scores = np.add(scores, -A * strategies[:, :, state])
    
    return A, scores, gains


def GameSimulationImitationTermal(S, M, N, Ncop, T, dynT):
    strategies = 2 * np.random.randint(2, size=(N, S, 2**M)) - 1
    scores = np.zeros((N, S))

    nodes_imitators = np.array(random.sample(range(N), Ncop))
    nodes_imitators.sort()
    nodes = []
    for i in range(N):
        if i not in nodes_imitators:
            nodes.append(i)
    nodes = np.array(nodes)

    A_t = np.zeros(T)
    G_t = np.zeros(T)
    GI_t = np.zeros(T)
    for t in range(T):
        who_now = np.random.rand(len(nodes_imitators))
        imitators_now = nodes_imitators[who_now<(1/dynT)]
        A, scores, gains = GameStepImitationTermal(S, M, N, Ncop, t,
                                        nodes, imitators_now, strategies, scores)
        A_t[t] = A
        G_t[t] = np.mean(gains[nodes])
        GI_t[t] = np.mean(gains[nodes_imitators])

    return A_t, G_t, GI_t

###Same for concrete networks, not complete graph
def ImitationMaximum(strategies, scores, S, imitators_now, counts, neighs):
    scores_max = np.argmax(scores, axis=1)
    scores_min_imitators = np.argmin(scores[imitators_now], axis=1)
    maxneighs = np.zeros(len(imitators_now)).astype(int)
    numstratneigh = np.zeros(len(imitators_now)).astype(int)
    for i in range(len(imitators_now)):
        node = imitators_now[i]
        maximoneigh = np.amax(scores[neighs[counts[node]:counts[node+1]]].flatten())
        iguales = np.argwhere(
            scores[neighs[counts[node]:counts[node+1]]].flatten() == maximoneigh).flatten()
        cual = np.random.choice(iguales)
        strat = cual%S
        neigh_con_maximo = neighs[counts[node]:counts[node+1]][int(cual/S)]
        maxneighs[i] = neigh_con_maximo
        numstratneigh[i] = strat
    copy_from = np.argwhere(scores[imitators_now, scores_max[imitators_now]] < scores[
        maxneighs, numstratneigh]).flatten()
    strategies[imitators_now[copy_from],scores_min_imitators[copy_from],:] = strategies[
                                        maxneighs[copy_from], numstratneigh[copy_from], :]
    scores[imitators_now[copy_from], scores_min_imitators[copy_from]] = 0
    return strategies, scores


def GameStepImitationTermalGraphs(S, M, N, Ncop, t, counts, neighs, 
                        nodes, imitators_now, strategies, scores):

    #DYNAMICS OF IMITATION
    if len(imitators_now) > 0:
        strategies, scores = ImitationMaximum(strategies, scores, S,
                                                imitators_now, counts, neighs)

    #DYNAMICS OF MINORITY GAME
    scores_max = np.argmax(scores, axis=1)
    scores_equal = np.argwhere((scores[:, 0] == scores[:, 1])).flatten()
    scores_max[scores_equal] = np.random.randint(S,size=len(scores_equal))
    
    state = np.random.randint(2**M)
    ai = strategies[np.arange(N),scores_max,state]
    
    A = np.sum(ai)
    gains = -A * ai
    scores = np.add(scores, -A * strategies[:, :, state])

    
    return A, scores, gains


def GameSimulationImitationTermalGraphs(S, M, N, Ncop, T, dynT, net, grado, p):
    strategies = 2 * np.random.randint(2, size=(N, S, 2**M)) - 1
    scores = np.zeros((N, S))

    if net == 0:
        counts, neighs = Neighbors_R(N, grado)
    if net == 1:
        counts, neighs = Neighbors_SmallWorld(N, grado, p)
    if net == 2:
        counts, neighs = Neighbors_BA(N, int(grado/2))

    nodes_imitators = np.array(random.sample(range(N), Ncop))
    nodes_imitators.sort()
    nodes = []
    for i in range(N):
        if i not in nodes_imitators:
            nodes.append(i)
    nodes = np.array(nodes)

    A_t = np.zeros(T)
    G_t = np.zeros(T)
    GI_t = np.zeros(T)
    for t in range(T):
        who_now = np.random.rand(len(nodes_imitators))
        imitators_now = nodes_imitators[who_now<(1/dynT)]

        A, scores, gains = GameStepImitationTermalGraphs(S, M, N, 
                                        Ncop, t, counts, neighs,
                                        nodes, imitators_now, strategies, scores,
                                        )
        A_t[t] = A
        G_t[t] = np.mean(gains[nodes])
        GI_t[t] = np.mean(gains[nodes_imitators])

    return A_t, G_t, GI_t


#--------#
###INFO###
#--------#

def GameStepImitationTermalINFO(S, M, N, Ncop, t, nodes, imitators_now,
                        strategies, scores, infos):

    #DYNAMICS OF IMITATION
    if len(imitators_now) > 0:
        strategies, scores = MaxImitationTermal(strategies, scores, N, S,
                                                        imitators_now)

    #DYNAMICS OF MINORITY GAME
    scores_max = np.argmax(scores, axis=1)
    scores_equal = np.argwhere((scores[:, 0] == scores[:, 1])).flatten()
    scores_max[scores_equal] = np.random.randint(S,size=len(scores_equal))
    
    state = np.random.randint(2**M)
    ai = strategies[np.arange(N),scores_max,state]
    
    A = np.sum(ai)
    scores = np.add(scores, -A * strategies[:, :, state])

    infos[state,t] = A
    
    return A, scores, infos


def GameSimulationImitationTermalINFO(S, M, N, Ncop, T, dynT):
    strategies = 2 * np.random.randint(2, size=(N, S, 2**M)) - 1
    scores = np.zeros((N, S))

    nodes_imitators = np.array(random.sample(range(N), Ncop))
    nodes_imitators.sort()
    nodes = []
    for i in range(N):
        if i not in nodes_imitators:
            nodes.append(i)
    nodes = np.array(nodes)

    A_t = np.zeros(T)
    infos = np.zeros((2**M,T))
    mean_infos = np.zeros(2**M)

    for t in range(T):
        who_now = np.random.rand(len(nodes_imitators))
        imitators_now = nodes_imitators[who_now<(1/dynT)]
        A, scores, infos = GameStepImitationTermalINFO(S, M, N, Ncop, t,
                                        nodes, imitators_now, strategies, scores, 
                                        infos)
        A_t[t] = A

    for m in range(2**M):
        inf = infos[m]
        info = inf[inf!=0]
        if len(info)>0:
            mean_infos[m] = np.mean(info)**2

    return A_t, mean_infos


def GameStepImitationTermalGraphsINFO(S, M, N, Ncop, t, counts, neighs, 
                        nodes, imitators_now, strategies, scores, infos):

    #DYNAMICS OF IMITATION
    if len(imitators_now) > 0:
        strategies, scores = ImitationMaximum(strategies, scores, S,
                                                imitators_now, counts, neighs)

    #DYNAMICS OF MINORITY GAME
    scores_max = np.argmax(scores, axis=1)
    scores_equal = np.argwhere((scores[:, 0] == scores[:, 1])).flatten()
    scores_max[scores_equal] = np.random.randint(S,size=len(scores_equal))
    
    state = np.random.randint(2**M)
    ai = strategies[np.arange(N),scores_max,state]
    
    A = np.sum(ai)
    scores = np.add(scores, -A * strategies[:, :, state])

    infos[state,t] = A
    
    return A, scores, infos


def GameSimulationImitationTermalGraphsINFO(S, M, N, Ncop, T, dynT, net, grado, p):
    strategies = 2 * np.random.randint(2, size=(N, S, 2**M)) - 1
    scores = np.zeros((N, S))

    if net == 0:
        counts, neighs = Neighbors_R(N, grado)
    if net == 1:
        counts, neighs = Neighbors_SmallWorld(N, grado, p)
    if net == 2:
        counts, neighs = Neighbors_BA(N, int(grado/2))

    nodes_imitators = np.array(random.sample(range(N), Ncop))
    nodes_imitators.sort()
    nodes = []
    for i in range(N):
        if i not in nodes_imitators:
            nodes.append(i)
    nodes = np.array(nodes)

    A_t = np.zeros(T)
    infos = np.zeros((2**M,T))
    mean_infos = np.zeros(2**M)

    for t in range(T):
        who_now = np.random.rand(len(nodes_imitators))
        imitators_now = nodes_imitators[who_now<(1/dynT)]

        A, scores, infos = GameStepImitationTermalGraphsINFO(S, M, N, 
                                        Ncop, t, counts, neighs,
                                        nodes, imitators_now, strategies, scores,
                                        infos)
        A_t[t] = A

    for m in range(2**M):
        inf = infos[m]
        info = inf[inf!=0]
        if len(info)>0:
            mean_infos[m] = np.mean(info)**2

    return A_t, mean_infos
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
from numba import jit


@jit(nopython=True)
def integer_state(history):
    """
    maps a list of m combinations of 1s and -1s to a unique number
    INPUT| a list of the last m outputs of the game
    OUTPUT| a unique integer
    """
    state = 0
    p = len(history)-1
    for i in history:
        if i == 1:
            state += 2**p
        p -= 1
    return state


@jit(nopython=True)
def minority(A):
    """
    Return the winner action
    """
    if A < 0:
        return 1
    elif A > 0:
        return -1
    else:
        return 2*random.randint(0, 1)-1


@jit(nopython=True)
def update_scores(N, S, A, strategies, state, scores, winner):
    for n in range(N):
        for s in range(S):
            if strategies[n][s][state] == winner:
                scores[n][s] += 1  # g[A(t)] = sign(A(t))
                #scores[n][s] += A
            else:
                scores[n][s] -= 1  # g[A(t)] = sign(A(t))
                #scores[n][s] += A
    return scores


def maximum_index(score):
    """
    Returns the index of the maximum score (if more than one max, selects randomly)
    """
    maximos = np.argwhere(score == np.amax(score)).flatten()
    return np.random.choice(maximos)

def set_strategies(S,N,M,strategies,c):
    for i in range(N):
        toss = np.random.rand(2**M)
        for j in range(2**M):
            if toss[j] > c:
                strategies[i,1,j] = - strategies[i,0,j]
            else:
                strategies[i,1,j] = strategies[i,0,j]
    return strategies

def one_game(N, S, strategies, history, scores):
    actions = np.zeros(N)
    state = integer_state(history)
    for i in range(N):
        index_strategy = maximum_index(scores[i])
        actions[i] = strategies[i][index_strategy][state]
    A = np.sum(actions)
    winner = minority(A)
    scores = update_scores(N, S, A, strategies, state, scores, winner)
    history = np.concatenate((history[1:], [winner]))
    return A, history, scores


def one_simulation(N, S, M, T, imprime=10**8):
    history = 2*np.random.randint(2, size=M)-1
    strategies = 2*np.random.randint(2, size=(N, S, 2**M))-1
    scores = np.zeros((N, S))
    attendances = np.zeros(T)
    times = np.zeros(T)
    meanA = np.zeros(T)  # to control the convergence
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A, history, scores = one_game(N, S, strategies, history, scores)
        times[t] = t
        attendances[t] = A
        meanA[t] = np.mean(attendances[:t+1])  # mean of the attendance so far
    return times, attendances, meanA



def one_game_frozen(N, S, strategies, history, scores):
    actions = np.zeros(N)
    state = integer_state(history)
    for i in range(N):
        index_strategy = maximum_index(scores[i])
        actions[i] = strategies[i][index_strategy][state]
    A = np.sum(actions)
    winner = minority(A)
    polarization_previous = scores[:, 0]-scores[:, 1]
    scores = update_scores(N, S, A, strategies, state, scores, winner)
    # print('scores')
    # print(scores)
    polarization_next = scores[:, 0]-scores[:, 1]
    #polarization_onegame = scores[:, 0]-scores[:, 1]
    polarization_onegame = polarization_next-polarization_previous
    # print('polarization')
    # print(polarization_onegame)
    history = np.concatenate((history[1:], [winner]))
    return A, history, scores, polarization_onegame


def one_simulation_frozen(N, S, M, T, imprime=10**8):
    # inizializing
    history = 2*np.random.randint(2, size=M)-1
    strategies = 2*np.random.randint(2, size=(N, S, 2**M))-1
    scores = np.zeros((N, S))
    attendances = np.zeros(T)
    times = np.zeros(T)
    polarization = np.zeros(N)
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('     t={}/{}'.format(t+1, T))
        A, history, scores, polarization_onegame = one_game_frozen(
            N, S, strategies, history, scores)
        times[t] = t
        attendances[t] = A
        polarization += polarization_onegame
    polarization /= T
    return times, attendances, np.abs(polarization)


def one_game_info(N, S, strategies, history, scores, information):
    actions = np.zeros(N)
    state = integer_state(history)
    for i in range(N):
        index_strategy = maximum_index(scores[i])
        actions[i] = strategies[i][index_strategy][state]
    A = np.sum(actions)
    winner = minority(A)
    information[state].append(winner)
    scores = update_scores(N, S, A, strategies, state, scores, winner)
    history = np.concatenate((history[1:], [winner]))
    return A, history, scores, information


def one_simulation_info(N, S, M, T, imprime=10**8):
    # inizializing
    history = 2*np.random.randint(2, size=M)-1
    strategies = 2*np.random.randint(2, size=(N, S, 2**M))-1
    scores = np.zeros((N, S))
    information = [[] for _ in range(2**M)]
    attendances = np.zeros(T)
    times = np.zeros(T)
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('     t={}/{}'.format(t+1, T))
        A, history, scores, information = one_game_info(
            N, S, strategies, history, scores, information)
        times[t] = t
        attendances[t] = A
    #print('information=\n', information)
    mean_informations = []
    for i in range(len(information)):
        asumar = information[i]
        if len(asumar) > 0:
            mean_informations.append(np.mean(asumar)**2)
        else:
            mean_informations.append(0)
    #information = [np.mean(information[i])**2 for i in range(len(information))]
    #print('informationmean=\n', information)
    return times, attendances, mean_informations


def one_game_variables(N, S, M, state, Omega, z_imu, Delta, pol):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    A = Omega[state] + np.sum(np.multiply(z_imu[:,state],si))
    X = -np.sign(A)
    pol = 2*X*z_imu[:,state]
    Delta = np.add(Delta, pol)
    state = int(np.mod((2*(state+1) + (X-1)/2),2**M)-1)  #el +1 (state + 1) y -1 al final 
                                    #son para que state vaya de 0 a 2**M-1
    if state == -1:
        state = 2**M-1
    return A, state, Delta, pol


def one_simulation_variables(N, S, M, T, imprime=10**8):
    state = np.random.randint(2**M)
    strategies = 2*np.random.randint(2, size=(N, S, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Omega = np.sum(w_imu,axis=0)
    scores = np.zeros((N, S))
    Delta = scores[:,0]-scores[:,1]
    attendances = np.zeros(T)
    times = np.zeros(T)
    polarization = np.zeros(N)
    pol = 0
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A, state, Delta, pol = one_game_variables(N, S, M, state, Omega, z_imu, Delta, pol)
        times[t] = t
        attendances[t] = A
        polarization += pol
    return times, attendances, np.abs(polarization)


def one_game_fake_memory(N, S, M, Omega, z_imu, Delta):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    state = np.random.randint(2**M) #the history state is drawn randomly each time now
    A = Omega[state] + np.sum(np.multiply(z_imu[:,state],si))
    X = -np.sign(A)
    pol = 2*X*z_imu[:,state]
    Delta = np.add(Delta, pol)

    return A, Delta

def one_simulation_fake_memory(N, S, M, T, imprime=10**8):
    strategies = 2*np.random.randint(2, size=(N, S, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Omega = np.sum(w_imu,axis=0)
    scores = np.zeros((N, S))
    Delta = scores[:,0]-scores[:,1]
    attendances = np.zeros(T)
    times = np.zeros(T)
    polarization = np.zeros(N)
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A, Delta = one_game_fake_memory(N, S, M, Omega, z_imu, Delta)
        times[t] = t
        attendances[t] = A
    return times, attendances


###market mechanicm

def one_game_specprods(Nspecs, Nprods, S, M, state, w_imu, z_imu, Delta_s, 
                                                strategies_prods, gain_specs, gain_prods):
    si_s = np.sign(Delta_s)
    indices_ceros = np.argwhere(si_s == 0).flatten()
    for i in indices_ceros:
        si_s[i] = 2*np.random.randint(2)-1
    #state = np.random.randint(2**M) #the history state is drawn randomly each time now
    ai_s = w_imu[:,state] + np.multiply(z_imu[:,state],si_s)
    ai_p = strategies_prods[:,state]
    A = np.sum(ai_s)+np.sum(ai_p)
    X = -np.sign(A)
    #gain_specs = np.add(gain_specs,-ai[:Nspecs]*A)
    gain_specs = -ai_s*A
    #gain_prods = np.add(gain_prods,-ai[Nspecs:]*A)
    gain_prods = -ai_p*A
    #pol = 2*X*z_imu[:,state]
    pol = -2*A*z_imu[:,state]
    Delta_s = np.add(Delta_s, pol)

    state = int(np.mod((2*(state+1) + (X-1)/2),2**M)-1)  #el +1 (state + 1) y -1 al final 
                                    #son para que state vaya de 0 a 2**M-1
    if state == -1:
        state = 2**M-1

    return A, state, Delta_s, gain_specs, gain_prods

def one_simulation_specprods(Nspecs, Nprods, S, M, T, c=0.5, imprime=10**8):
    strategies_specs = 2*np.random.randint(2, size=(Nspecs, S, 2**M))-1
    if c != 0.5:
        strategies_specs = set_strategies(S,Nspecs,M,strategies_specs,c)
    #strategies_prods = 2*np.random.randint(2, size=(Nprods, S, 2**M))-1
    strategies_prods = 2*np.random.randint(2, size=(Nprods, 2**M))-1
    #strategies_prods = set_strategies(S,Nprods,M,strategies_prods,1)
    #strategies = np.concatenate((strategies_specs,strategies_prods))
    #N = Nspecs+Nprods
    #w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    #z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    w_imu = (strategies_specs[:,0,:]+strategies_specs[:,1,:])/2
    z_imu = (strategies_specs[:,0,:]-strategies_specs[:,1,:])/2
    #Omega = np.sum(w_imu,axis=0)
    #scores = np.zeros((N, S))
    scores_s = np.zeros((Nspecs,2))
    Delta_s = scores_s[:,0]-scores_s[:,1]
    
    attendances = np.zeros(T)
    times = np.zeros(T)
    mean_gain_specs = np.zeros(T)
    mean_gain_prods = np.zeros(T)

    gain_specs = np.zeros(Nspecs)
    gain_prods = np.zeros(Nprods)
    # simulation
    state = np.random.randint(2**M) #initial random state
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A, state, Delta_s, gain_specs, gain_prods = one_game_specprods(Nspecs, Nprods, S, M,  
                                                state, w_imu, z_imu, Delta_s, 
                                                strategies_prods, gain_specs, gain_prods)
        times[t] = t
        attendances[t] = A
        mean_gain_specs[t] = np.mean(gain_specs)
        mean_gain_prods[t] = np.mean(gain_prods)

    return times, attendances, mean_gain_specs, mean_gain_prods


def one_game_grandcanon(Nspecs, Nprods, M, state, scores_s, strategies_specs,
                                                strategies_prods):
    #state = np.random.randint(2**M) #the history state is drawn randomly each time now

    scores_max = np.argmax(scores_s,axis=1)
    scores_equal = np.argwhere((scores_s[:,0] == scores_s[:,1])).flatten()
    scores_nogain = np.argwhere((scores_s[:,0]<1) & (scores_s[:,1]<1)).flatten()

    print('state = ', state)
    ai_s = np.zeros(Nspecs)
    if len(scores_max)>0:
        for n in range(Nspecs): #choose the action of the strategy with maximum score
            ai_s[n] = strategies_specs[n,scores_max[n],state]
    if len(scores_equal)>0:
        for n in scores_equal: #if two maximum, choose randomly one of the two strategies
            ai_s[n] = np.random.choice(strategies_specs[n,:,state])
    if len(scores_nogain)>0:
        ai_s[scores_nogain] = 0 #if the two scores are negative, the player do not play

    ai_p = strategies_prods[:,state]
    A = np.sum(ai_s) + np.sum(ai_p)
    jugados = ai_s[ai_s!=0]
    gain_specs = -jugados*A
    gain_prods = -ai_p*A
    scores_s = np.add(scores_s,-A*strategies_specs[:,:,state])
    state = int(np.mod((2*(state+1) + (-np.sign(A)-1)/2),2**M)-1)  #el +1 (state + 1) y -1 al final 
                                    #son para que state vaya de 0 a 2**M-1
    if state == -1:
        state = 2**M-1
    s_played = len(jugados)
    if s_played>0:
        gain_specs = np.sum(gain_specs)/s_played

    return A, state, scores_s, gain_specs, np.mean(gain_prods), s_played

def one_simulation_grandcanon(Nspecs, Nprods, S, M, T, c=0.5, imprime=10**8):
    strategies_specs = 2*np.random.randint(2, size=(Nspecs, S, 2**M))-1 #2 strategies speculators
    if c != 0.5:
        strategies_specs = set_strategies(S,Nspecs,M,strategies_specs,c)
    strategies_prods = 2*np.random.randint(2, size=(Nprods, 2**M))-1    #1 strategy producers

    scores_s = np.ones((Nspecs,S)) #scores speculators
    
    attendances = np.zeros(T)
    times = np.zeros(T)
    mean_gain_specs = np.zeros(T)
    mean_gain_prods = np.zeros(T)
    s_played_t = np.zeros(T)

    # simulation
    state = np.random.randint(2**M) #initial random state
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A, state, scores_s, gain_specs, gain_prods, s_played = one_game_grandcanon(Nspecs, Nprods, M,  
                                                state, scores_s, strategies_specs, strategies_prods)
        times[t] = t
        attendances[t] = A
        mean_gain_specs[t] = gain_specs
        mean_gain_prods[t] = gain_prods
        s_played_t[t] = s_played
        
    return times, attendances, mean_gain_specs, mean_gain_prods, s_played_t


def one_game_spriv(Nspecs, Nprods, M, state, scores_s, scores_priv,
                                        strategies_specs, strategies_prods, strategy_priv):
    scores_max = np.argmax(scores_s,axis=1)
    scores_equal = np.argwhere((scores_s[:,0] == scores_s[:,1])).flatten()
    ai_s = np.zeros(Nspecs)
    if len(scores_max)>0:
        for n in range(Nspecs): #choose the action of the strategy with maximum score
            ai_s[n] = strategies_specs[n,scores_max[n],state]
    if len(scores_equal)>0:
        for n in scores_equal: #if two maximum, choose randomly one of the two strategies
            ai_s[n] = np.random.choice(strategies_specs[n,:,state])

    scorepriv_maxval = np.amax(scores_priv)
    scorepriv_maxreps = np.argwhere(scores_priv == scorepriv_maxval).flatten()
    used = np.random.choice(scorepriv_maxreps)
    apriv = strategy_priv[used,state]

    ai_p = strategies_prods[:,state]

    A = np.sum(ai_s) + np.sum(ai_p) + apriv

    scores_s = np.add(scores_s,-A*strategies_specs[:,:,state])
    scores_priv = np.add(scores_priv,-A*strategy_priv[:,state])

    gain_priv = -A*apriv

    state = int(np.mod((2*(state+1) + (-np.sign(A)-1)/2),2**M)-1)  #el +1 (state + 1) y -1 al final 
                                    #son para que state vaya de 0 a 2**M-1
    if state == -1:
        state = 2**M-1

    return A, state, scores_s, scores_priv, gain_priv, used

def one_simulation_spriv(Nspecs, Nprods, S, Sprima, M, T, c=0.5, imprime=10**8):
    strategies_specs = 2*np.random.randint(2, size=(Nspecs, S, 2**M))-1
    if c != 0.5:
        strategies_specs = set_strategies(S,Nspecs,M,strategies_specs,c)
    w_imu = (strategies_specs[:,0,:]+strategies_specs[:,1,:])/2
    z_imu = (strategies_specs[:,0,:]-strategies_specs[:,1,:])/2
    scores_s = np.zeros((Nspecs,2))
    Delta_s = scores_s[:,0]-scores_s[:,1]

    strategies_prods = 2*np.random.randint(2, size=(Nprods, 2**M))-1
    strategy_priv = 2*np.random.randint(2,size=(Sprima,2**M))
    strategy_virt = 2*np.random.randint(2,size=(Sprima,2**M))
    scores_priv = np.zeros(Sprima)
    scores_virt = np.zeros(Sprima)
    
    attendances = np.zeros(T)
    times = np.zeros(T)
    gain = np.zeros(T)
    strategies_used = []

    state = np.random.randint(2**M) #initial random state
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A, state, scores_s, scores_priv, gain_priv, used = one_game_spriv(Nspecs, Nprods, 
                                                M, state,
                                                scores_s, scores_priv,
                                                strategies_specs, strategies_prods, strategy_priv)
        times[t] = t
        attendances[t] = A
        gain[t] = gain_priv
        if used not in strategies_used:
            strategies_used.append(used)

    return times, attendances, gain, len(strategies_used)


def one_game_spy(N, M, NB, state, w_imu, z_imu, Delta, strategy, scorespy):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    zstate = z_imu[:,state]
    ai_si = w_imu[:,state] + np.multiply(zstate,si)

    sb = int((np.sign(np.sum(ai_si[:NB]))+1)/2) #set scores U+ son el 0, set scores U- son el 1
    if scorespy[sb,0] == scorespy[sb,1]:
        s_spy = np.random.randint(2) # o la 0 o la 1 dentro del set elegido
    if scorespy[sb,0] != scorespy[sb,1]:
        s_spy = np.argmax(scorespy[sb])

    a_spy = strategy[sb,s_spy,state]

    A = np.sum(ai_si) + a_spy

    gains = -ai_si*A
    gainspy = -a_spy*A

    Delta = np.add(Delta, -2*A*zstate)
    scorespy[sb] = np.add(scorespy[sb],-A*strategy[sb,:,state])

    state = int(np.mod((2*(state+1) + (-np.sign(A)-1)/2),2**M)-1)
    if state == -1:
        state = 2**M-1

    return A, state, Delta, scorespy, gains, gainspy


def one_simulation_spy(N, M, NB, T, imprime=10**8):
    #normal speculators
    strategies = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Delta = np.zeros(N)

    #the spy has two sets of two strategies
    strategy = 2*np.random.randint(2, size=(2, 2, 2**M))-1
    scorespy = np.zeros((2,2))

    #to record
    gain_s = np.zeros(T)
    gain_spy = np.zeros(T)

    # simulation
    state = np.random.randint(2**M)
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        state,Delta,scorespy,gains,gainspy = one_game_spy(N, M, NB, 
                                                            state,
                                                            w_imu, z_imu, Delta,
                                                            strategy, scorespy)
        gain_s[t] = np.mean(gains)
        gain_spy[t] = gainspy

    return np.mean(gain_s), np.mean(gain_spy)
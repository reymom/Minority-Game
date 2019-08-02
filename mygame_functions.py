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
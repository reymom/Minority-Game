import numpy as np

#-----------------------------------------------------#
#----------------Some useful functions----------------#
#-----------------------------------------------------#


# If you want the whole sequence of last winning groups, #
#  you transform to an integer with that. Nevertheless,  #
#        you normally update the integer directly.       #

def integer_state(history):
    """
    Maps a list of m combinations of 1s and -1s to a unique integer
    INPUT | history: a list of the last m outputs of the game
    OUTPUT| state:  unique integer
    """
    state = 0
    p = len(history)-1
    for i in history:
        if i == 1:
            state += 2**p
        p -= 1
    return state


#   If you want to draw strategies with some correlation    #
#  you need to use this function and pass the correlation c #

def set_strategies(S,N,M,strategies,c):
    """
    Gives strategies with certain correlation c

    INPUT | strategies: the vector of the random already defined strategies
          | c         : the correlation. c=0 all strategies equal, c=1 contraries

    OUTPUT| the vector strategies with the correlation desired
    """
    for i in range(N):
        toss = np.random.rand(2**M)
        for j in range(2**M):
            if toss[j] > c:
                strategies[i,1,j] = - strategies[i,0,j]
            else:
                strategies[i,1,j] = strategies[i,0,j]
    return strategies


#-----------------------------------------------------#
#----------------GAMES AND SIMULATIONS----------------#
#-----------------------------------------------------#


            #-----------------------------#
            #----------BASICS MG----------#
            #-----------------------------#


def GameStep(N, M, scores, strategies):
    state = np.random.randint(2**M) #the history state is drawn randomly each time now
    scores_max = np.argmax(scores,axis=1)
    scores_equal = np.argwhere((scores[:,0] == scores[:,1])).flatten()
    ai = np.zeros(N)
    for nm in range(N): #choose the action of the strategy with maximum score
        ai[nm] = strategies[nm,scores_max[nm],state]
    if len(scores_equal)>0:
        for ne in scores_equal: #if two maximum, choose randomly one of the strategies
            ai[ne] = np.random.choice(strategies[ne,:,state])
    A = np.sum(ai)
    scores = np.add(scores,-A*strategies[:,:,state])
    return A, scores

def GameSimulation(N, S, M, T, imprime=10**8):
    strategies = 2*np.random.randint(2, size=(N, S, 2**M))-1
    scores = np.zeros((N,S))
    A_t = np.zeros(T)
    for t in range(T):
        if (t+1) % imprime == 0:
            print('             t = {}/{}'.format(t+1, T))
        A, scores = GameStep(N, M, scores, strategies)
        A_t[t] = A
    return A_t


def GameStepVariables(N, S, M, state, Omega, z_imu, Delta, pol):
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

def GameSimulationVariables(N, S, M, T, imprime=10**8):
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
        A, state, Delta, pol = GameStepVariables(N, S, M, state, Omega, z_imu, Delta, pol)
        times[t] = t
        attendances[t] = A
        polarization += pol
    return times, attendances, np.abs(polarization)

def GameStepVirtualGain(M, w_imu, z_imu, Delta, w_imu_v, z_imu_v, Delta_v):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    state = np.random.randint(2**M)
    zstate = z_imu[:,state]
    ai_si = w_imu[:,state] + np.multiply(zstate, si)

    si_v = np.sign(Delta_v)
    indices_ceros_v = np.argwhere(si_v == 0).flatten()
    for i in indices_ceros_v:
        si_v[i] = 2*np.random.randint(2)-1
    zstate_v = z_imu_v[:,state]
    ai_si_v = w_imu_v[:,state] + np.multiply(zstate_v, si_v)

    A = np.sum(ai_si)

    rgain = -A*ai_si
    vgain = -A*ai_si_v

    Delta = np.add(Delta, -2*A*zstate)
    Delta_v = np.add(Delta_v, -2*A*zstate_v)

    return Delta, Delta_v, rgain, vgain

def GameSimulationVirtualGain(N, M, T):
    #strategies for active players and passive (measuring virtual scores)
    strategies = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Delta = np.zeros(N)

    strategies_v = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu_v = (strategies_v[:,0,:]+strategies_v[:,1,:])/2
    z_imu_v = (strategies_v[:,0,:]-strategies_v[:,1,:])/2
    Delta_v = np.zeros(N)

    real_gains = np.zeros(T)
    virt_gains = np.zeros(T)
    maxreal_delta = np.zeros(T)
    maxvirt_delta = np.zeros(T)
    # simulation
    for t in range(T):
        Delta, Delta_v, rgain, vgain = GameStepVirtualGain(M, w_imu, z_imu, Delta, 
                                    w_imu_v, z_imu_v, Delta_v)

    real_gains[t] = np.mean(rgain)
    virt_gains[t] = np.mean(vgain)
    maxreal_delta[t] = np.mean(Delta)
    maxvirt_delta[t] = np.mean(Delta_v)

    return real_gains, virt_gains, maxreal_delta, maxvirt_delta


def GameStepInfo(M, state, w_imu, z_imu, Delta, infos):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    zstate = z_imu[:,state]
    ai_si = w_imu[:,state] + np.multiply(zstate, si)

    A = np.sum(ai_si)

    win = -np.sign(A)
    Delta = np.add(Delta, 2*win*zstate)

    infos[state].append(win)

    state = int(np.mod((2*(state+1) + (win-1)/2),2**M)-1)
    if state == -1:
        state = 2**M-1

    return infos, state, Delta

def GameSimulationInfo(N, M, T, imprime=10**8):
    #strategies, its variables and initial scores
    strategies = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Delta = np.zeros(N)

    # to record
    infos = [[] for _ in range(2**M)]

    # simulation
    state = np.random.randint(2**M)
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        infos, state, Delta = GameStepInfo(M, state, w_imu, z_imu,
                                                    Delta, infos)
    mean_infos = []
    for infostate in infos:
        if len(infostate) > 0:
            mean_infos.append(np.mean(infostate)**2)
        else:
            mean_infos.append(0)
    mean_infos2 = []
    for infostate in infos:
        if len(infostate) > 0:
            mean_infos2.append(np.mean(infostate)**2)

    #if you want to do the histograms for one simulation, return infos
    return np.mean(mean_infos), np.mean(mean_infos2)


def GameStepInfoCorr(M, state, w_imu, z_imu, Delta, infos, t):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    zstate = z_imu[:,state]
    ai_si = w_imu[:,state] + np.multiply(zstate, si)

    A = np.sum(ai_si)

    Delta = np.add(Delta, -2*A*zstate)

    win = -np.sign(A)
    infos[state,t] = win

    state = int(np.mod((2*(state+1) + (win-1)/2),2**M)-1)
    if state == -1:
        state = 2**M-1

    return infos, state, Delta

def GameSimulationInfoCorr(N, M, T, imprime=10**8):
    #strategies, its variables and initial scores
    strategies = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Delta = np.zeros(N)

    # to record
    infos = np.zeros((2**M,T))

    # simulation
    state = np.random.randint(2**M)
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        infos, state, Delta = GameStepInfoCorr(M, state, w_imu, z_imu,
                                                    Delta, infos, t)
    return infos

def GameStepPolariz(M, w_imu, z_imu, Delta):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    state = np.random.randint(2**M)
    zstate = z_imu[:,state]
    ai_si = w_imu[:,state] + np.multiply(zstate, si)
    A = np.sum(ai_si)
    Delta = np.add(Delta, -2*np.sign(A)*zstate)
    return Delta

def GameSimulationPolariz(N, M, T, imprime=10**8):
    #strategies, its variables and initial scores
    strategies = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Delta = np.zeros(N)

    polarization = np.zeros((N,T))
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        Delta = GameStepPolariz(M, w_imu, z_imu, Delta)

        polarization[:,t] = Delta

    return polarization #, np.mean(polarization,axis=1)


            #----------------------------#
            #------MARKET MECHANISM------#
            #----------------------------#

def GameStepProducers(M, w_imu, z_imu, Delta_s, 
                            strategies_p):
    si_s = np.sign(Delta_s)
    indices_ceros = np.argwhere(si_s == 0).flatten()
    for i in indices_ceros:
        si_s[i] = 2*np.random.randint(2)-1
    state = np.random.randint(2**M)

    zstate = z_imu[:,state]

    ai_s = w_imu[:,state] + np.multiply(zstate,si_s)
    ai_p = strategies_p[:,state]

    A = np.sum(ai_s)+np.sum(ai_p)

    gain_s = -ai_s*A
    gain_p = -ai_p*A

    Delta_s = np.add(Delta_s, -2*A*zstate)

    return Delta_s, gain_s, gain_p

def GameSimulationProducers(Ns, Np, M, T, c=0.5, imprime=10**8):
    strategies_s = 2*np.random.randint(2, size=(Ns, 2, 2**M))-1
    if c != 0.5:
        strategies_s = set_strategies(2,Ns,M,strategies_s,c)
    w_imu = (strategies_s[:,0,:]+strategies_s[:,1,:])/2
    z_imu = (strategies_s[:,0,:]-strategies_s[:,1,:])/2
    Delta_s = np.zeros(Ns)

    strategies_p = 2*np.random.randint(2, size=(Np, 2**M))-1
    
    meangain_s = np.zeros(T)
    meangain_p = np.zeros(T)
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        Delta_s, gain_s, gain_p = GameStepProducers(M,  
                                                w_imu, z_imu, Delta_s, 
                                                strategies_p)
        meangain_s[t] = np.mean(gain_s)
        meangain_p[t] = np.mean(gain_p)

    return meangain_s, meangain_p


def GameStepGC(Ns, M, scores_s, strategies_s, strategies_p):
    scores_max = np.argmax(scores_s,axis=1)
    scores_equal = np.argwhere((scores_s[:,0] == scores_s[:,1])).flatten()
    scores_nogain = np.argwhere((scores_s[:,0]<0) & (scores_s[:,1]<0)).flatten()

    state = np.random.randint(2**M)
    ai_s = np.zeros(Ns)
    for nm in range(Ns): #choose the action of the strategy with maximum score
        ai_s[nm] = strategies_s[nm,scores_max[nm],state]
    if len(scores_equal)>0:
        for ne in scores_equal: #if two maximum, choose randomly one of the two strategies
            ai_s[ne] = np.random.choice(strategies_s[ne,:,state])
    if len(scores_nogain)>0:
        ai_s[scores_nogain] = 0 #if the two scores are negative, the player do not play
    nplayers = len(ai_s[ai_s!=0])
    
    ai_p = strategies_p[:,state]

    A = np.sum(ai_s)+np.sum(ai_p)
    if A == 0:
        A+=2*np.random.randint(1)-1
    gain_s = -ai_s*A
    gain_p = -ai_p*A

    scores_s = np.add(scores_s, -A*strategies_s[:,:,state])

    return A, gain_s, gain_p, scores_s, nplayers

def GameSimulationGC(Ns, Np, M, T, c=0.5):
    strategies_s = 2*np.random.randint(2, size=(Ns, 2, 2**M))-1
    #if c != 0.5:
    #    strategies_s = set_strategies(2,Ns,M,strategies_s,c)
    scores_s = np.zeros((Ns,2))

    strategies_p = 2*np.random.randint(2, size=(Np, 2**M))-1

    A_t = np.zeros(T)
    meangain_s = np.zeros(T)
    meangain_p = np.zeros(T)
    nplayers_s = np.zeros(T)

    # simulation
    for t in range(T):
        A, gain_s, gain_p, scores_s, nplayers = GameStepGC(Ns, M,
                                            scores_s, strategies_s, strategies_p)
        A_t[t] = A
        meangain_s[t] = np.mean(gain_s)
        meangain_p[t] = np.mean(gain_p)
        nplayers_s[t] = nplayers

    return A_t, meangain_s, meangain_p, nplayers_s


def one_game_noise(N, Nnoise, S, M, Omega, z_imu, Delta):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    state = np.random.randint(2**M) #the history state is drawn randomly each time now
    anoise = 2*np.random.randint(2,size=(Nnoise))-1
    A = Omega[state] + np.sum(np.multiply(z_imu[:,state],si)) + np.sum(anoise)
    X = -np.sign(A)
    pol = 2*X*z_imu[:,state]
    Delta = np.add(Delta, pol)

    return A, Delta

def one_simulation_noise(N, Nnoise, S, M, T, imprime=10**8):
    strategies = 2*np.random.randint(2, size=(N, S, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Omega = np.sum(w_imu,axis=0)
    scores = np.zeros((N, S))
    Delta = scores[:,0]-scores[:,1]
    attendances = np.zeros(T)
    times = np.zeros(T)
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A, Delta = one_game_noise(N, Nnoise, S, M, Omega, z_imu, Delta)
        times[t] = t
        attendances[t] = A
    return times, attendances
    

def GameStepSPriv(M, w_imu, z_imu, Delta_s, A_p, strategies_priv,
                            scores_priv):
    si_s = np.sign(Delta_s)
    indices_ceros = np.argwhere(si_s == 0).flatten()
    for i in indices_ceros:
        si_s[i] = 2*np.random.randint(2)-1
    state = np.random.randint(2**M)
    zstate = z_imu[:,state]
    ai_s = w_imu[:,state] + np.multiply(zstate,si_s)

    score_maxval = np.amax(scores_priv)
    score_maxreps = np.argwhere(scores_priv == score_maxval).flatten()
    used = np.random.choice(score_maxreps)
    apriv = strategies_priv[used,state]

    A = np.sum(ai_s)+ A_p[state] #+ apriv virtual gain if #
    gain = -A*apriv
    Delta_s = np.add(Delta_s, -2*A*zstate)
    scores_priv = np.add(scores_priv,-A*strategies_priv[:,state])

    #infos[state].append(-A)

    return Delta_s, scores_priv, gain, used

def GameSimulationSPriv(Ns, Np, M, SP, T, imprime=10**8):
    strategies_s = 2*np.random.randint(2, size=(Ns, 2, 2**M))-1
    w_imu = (strategies_s[:,0,:]+strategies_s[:,1,:])/2
    z_imu = (strategies_s[:,0,:]-strategies_s[:,1,:])/2
    Delta_s = np.zeros(Ns)

    strategies_p = 2*np.random.randint(2, size=(Np, 2**M))-1
    A_p = np.sum(strategies_p, axis = 0)

    strategies_priv = 2*np.random.randint(2,size=(SP,2**M))-1
    scores_priv = np.zeros(SP)

    gain_sp = np.zeros(T)
    used_sp = np.zeros(SP)
    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        Delta_s, scores_priv, gain, used = GameStepSPriv(M,  
                                                w_imu, z_imu, Delta_s, 
                                                A_p, strategies_priv,
                                                scores_priv)
        gain_sp[t] = gain
        used_sp[used] += 1

    return gain_sp, len(used_sp[used_sp > 10])


def one_game_morememo(N, M, mp, state, statem, w_imu, w_imum, z_imu, z_imum, Delta, Deltam, infos, infom):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    zstate = z_imu[:,state]
    ai_si = w_imu[:,state] + np.multiply(zstate,si)

    sim = np.sign(Deltam)
    if sim == 0:
        sim = 2*np.random.randint(2)-1
    zstatem = z_imum[statem]
    a_m = w_imum[statem] + zstatem*sim

    A = np.sum(ai_si) + a_m

    gains = -ai_si*A
    gainm = -a_m*A

    Delta = np.add(Delta, -2*A*zstate)
    Deltam -= 2*A*zstatem

    infom[statem].append(A)
    infos[state].append(A)

    win = -np.sign(A)
    state = int(np.mod((2*(state+1) + (win-1)/2),2**M)-1)
    if state == -1:
        state = 2**M-1
    statem = int(np.mod((2*(statem+1) + (win-1)/2),2**mp)-1)
    if statem == -1:
        statem = 2**mp-1

    return A, infos, infom, state, statem, Delta, Deltam, gains, gainm


def one_simulation_morememo(N, M, mp, T, imprime=10**8):
    strategies = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Delta = np.zeros(N)

    strategy = 2*np.random.randint(2, size=(2, 2**mp))-1
    w_imum = (strategy[0,:]+strategy[1,:])/2
    z_imum = (strategy[0,:]-strategy[1,:])/2
    Deltam = 0

    #to record
    atts = np.zeros(T)
    times = np.zeros(T)
    gain_s = np.zeros(T)
    gain_m = np.zeros(T)

    infos = [[] for _ in range(2**M)]
    infom = [[] for _ in range(2**mp)]

    # simulation, with initial state
    initial_hist = 2*np.random.randint(2,size=mp)-1
    state = integer_state(initial_hist[(mp-M):])
    statem = integer_state(initial_hist)

    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        A,infos,infom,state,statem,Delta,Deltam,gains,gainm = one_game_morememo(N, M, mp, 
                                                            state, statem,
                                                            w_imu, w_imum, z_imu, z_imum,
                                                            Delta, Deltam, infos, infom)
        times[t] = t
        atts[t] = A
        gain_s[t] = np.mean(gains)
        gain_m[t] = gainm

    mean_infos = []
    for infostate in infos:
        if len(infostate) > 0:
            mean_infos.append(np.mean(infostate)**2)
        else:
            mean_infos.append(0)
    
    mean_info = []
    for infostate in infom:
        if len(infostate) > 0:
            mean_info.append(np.mean(infostate)**2)
        else:
            mean_info.append(0)

    return times, np.mean(mean_infos)/N, np.mean(mean_info)/N, np.mean(gain_s), np.mean(gain_m)

def GameStepSpy(M, NB, w_imu, z_imu, Delta, strategy, scorespy):
    si = np.sign(Delta)
    indices_ceros = np.argwhere(si == 0).flatten()
    for i in indices_ceros:
        si[i] = 2*np.random.randint(2)-1
    state = np.random.randint(2**M)
    zstate = z_imu[:,state]
    ai_si = w_imu[:,state] + np.multiply(zstate,si)

    sb = int((np.sign(np.sum(ai_si[:NB]))+1)/2) #2 scores, for output of spied +1 or -1
    if scorespy[sb,0] == scorespy[sb,1]:
        s_spy = np.random.randint(2) # o la 0 o la 1 dentro del set elegido
    if scorespy[sb,0] != scorespy[sb,1]:
        s_spy = np.argmax(scorespy[sb])

    a_spy = strategy[s_spy,state]

    A = np.sum(ai_si) + a_spy

    gains = -ai_si*A
    gainspy = -a_spy*A

    Delta = np.add(Delta, -2*A*zstate)
    scorespy[sb] = np.add(scorespy[sb],-A*strategy[:,state])


    return Delta, scorespy, gains, gainspy


def GameSimulationSpy(N, M, NB, T, imprime=10**8):
    #normal speculators
    strategies = 2*np.random.randint(2, size=(N, 2, 2**M))-1
    w_imu = (strategies[:,0,:]+strategies[:,1,:])/2
    z_imu = (strategies[:,0,:]-strategies[:,1,:])/2
    Delta = np.zeros(N)

    strategy = 2*np.random.randint(2, size=(2, 2**M))-1
    #two scores for each strategy
    scorespy = np.zeros((2,2))

    #to record
    gain_s = np.zeros(T)
    gain_spy = np.zeros(T)

    # simulation
    for t in range(T):
        if (t+1) % imprime == 0:
            print('         t={}/{}'.format(t+1, T))
        Delta, scorespy, gains, gainspy = GameStepSpy(M, NB, w_imu, z_imu, Delta,
                                                    strategy, scorespy)
        gain_s[t] = np.mean(gains)
        gain_spy[t] = gainspy

    return np.mean(gain_s), np.mean(gain_spy)

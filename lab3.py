import numpy as np

# ===================== 1. INITIALISE MRP =====================

def init_mrp(gamma=0.5):
    """
    Initialize the Markov Reward Process:
    S: states
    R: reward vector
    P: transition matrix
    gamma: discount factor
    """
    S=['c1','c2','c3','pass','rest','tv','sleep']
    R=np.array([-2,-2,-2,+10,+1,-1,0])
    P=np.array([
        [0.0,0.5,0.0,0.0,0.0,0.5,0.0],
        [0.0,0.0,0.8,0.0,0.0,0.0,0.2],
        [0.0,0.0,0.0,0.6,0.4,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,1.0],
        [0.2,0.4,0.4,0.0,0.0,0.0,0.0],
        [0.1,0.0,0.0,0.0,0.0,0.9,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,1.0]
    ])
    assert(np.all(np.sum(P,axis=1)==1))
    return S,R,P,gamma

# ===================== 2. SAMPLE ONE EPISODE =====================

def sample_episode(P,S,start_state_idx=0,terminal_state='sleep',log=True):
    """
    Sample an episode from the MRP starting at S[start_state_idx]
    until terminal_state (e.g., 'sleep') is reached.
    Returns: np.array of state indices.
    """
    s=start_state_idx
    episode=[s]
    print_str=S[s]+', '
    while S[episode[-1]]!=terminal_state:
        next_s=np.random.choice(len(P),1,p=P[episode[-1]])[0]
        episode.append(next_s)
        print_str+=S[next_s]+', '
    if log:
        print(print_str)
    return np.array(episode)

# ===================== 3. COMPUTE DISCOUNTED RETURN G_t =====================

def compute_return(episode,R,gamma):
    """
    Compute discounted return G_t for a given episode.
    G_t=sum_{k=0..T} gamma^k * R[s_k]
    """
    episode_rewards=R[episode]
    G_t=0.0
    for k in range(len(episode)):
        G_t+=gamma**k*episode_rewards[k]
    return G_t

def debug_return(episode,R,gamma):
    """
    Same as compute_return but prints intermediate values (for explanation).
    """
    episode_rewards=R[episode]
    G_t=0.0
    for k in range(len(episode)):
        G_t+=gamma**k*episode_rewards[k]
        print("step",k,"state_idx",episode[k],"reward",episode_rewards[k],
              "gamma^k={:.4f}".format(gamma**k),
              "G_t={:.4f}".format(G_t))
    return G_t

# ===================== 4. MONTE CARLO VALUE ESTIMATION =====================

def mc_value_estimation(S,R,P,gamma,num_episodes=2000,log_interval=100):
    """
    Monte Carlo estimation of value function V(s) for all states.
    For each episode index i:
      for each state s:
        - sample an episode starting from s
        - compute G_t
        - accumulate into V[s]
    Returns: V_hat (Monte Carlo estimate of V)
    """
    n_states=len(P)
    V=np.zeros(n_states)

    for i in range(num_episodes):
        for s in range(n_states):
            episode=sample_episode(P,S,start_state_idx=s,log=False)
            G_t=compute_return(episode,R,gamma)
            V[s]+=G_t
        if (i+1)%log_interval==0:
            np.set_printoptions(precision=2,suppress=True)
            print("After",i+1,"episodes, V estimate:")
            print(V/(i+1))
    V=V/num_episodes
    return V

# ===================== 5. EXACT VALUE VIA BELLMAN EQUATION =====================

def bellman_value(P,R,gamma):
    """
    Solve (I - gamma*P)V = R  =>  V = (I - gamma*P)^(-1) R
    Returns exact value function vector V.
    """
    n=len(P)
    I=np.identity(n)
    V=np.linalg.solve(I-gamma*P,R)
    return V

# ===================== 6. EXPERIMENT HELPERS =====================

def change_gamma(old_gamma,new_gamma):
    """
    Just a helper to make it clear you're changing gamma.
    """
    print("gamma changed from",old_gamma,"to",new_gamma)
    return new_gamma

def change_reward_tv(R,new_reward_tv,state_names):
    """
    Change reward of 'tv' state and return new R.
    """
    R_new=R.copy()
    tv_idx=state_names.index('tv')
    R_new[tv_idx]=new_reward_tv
    print("Reward for 'tv' changed to",new_reward_tv)
    return R_new

def add_state_exercise(S,R,P,gamma,
                       new_transitions_from_exercise,
                       new_rewards_exercise):
    """
    Optional advanced: add a new state 'exercise'.
    - new_transitions_from_exercise: list/array of probs to all old+new states (len n+1)
    - new_rewards_exercise: scalar reward for 'exercise'
    """
    S_new=S+['exercise']
    n=len(P)
    # Extend reward vector
    R_new=np.append(R,new_rewards_exercise)
    # Extend P with new row and column
    P_new=np.zeros((n+1,n+1))
    P_new[:n,:n]=P
    # transitions from exercise
    P_new[n,:]=new_transitions_from_exercise
    # transitions TO exercise (optional, here default 0 already)
    assert(np.all(np.sum(P_new,axis=1)==1))
    return S_new,R_new,P_new,gamma

# ===================== 7. QUICK DEMO (RUN TO SEE EVERYTHING) =====================

if __name__=="__main__":
    # init
    S,R,P,gamma=init_mrp(gamma=0.5)

    # --- Sample a few episodes ---
    print("first sample:")
    ep1=sample_episode(P,S,start_state_idx=0)
    print("\nsecond sample:")
    ep2=sample_episode(P,S,start_state_idx=0)
    print("\nthird sample:")
    ep3=sample_episode(P,S,start_state_idx=0)

    # --- Compute return for one episode with debug ---
    print("\nDebugging return for a fresh episode:")
    ep=sample_episode(P,S,start_state_idx=0,log=True)
    G_t=debug_return(ep,R,gamma)
    print("Final G_t:",G_t)

    # --- Monte Carlo estimation ---
    print("\nMonte Carlo estimation of V(s):")
    V_mc=mc_value_estimation(S,R,P,gamma,num_episodes=2000,log_interval=500)
    print("\nFinal Monte Carlo V:")
    print(V_mc)

    # --- Exact Bellman solution ---
    print("\nExact V via Bellman equation:")
    V_exact=bellman_value(P,R,gamma)
    print(V_exact)

    # --- Compare MC vs Exact ---
    print("\nDifference (MC - Exact):")
    print(V_mc-V_exact)

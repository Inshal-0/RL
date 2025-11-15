import gymnasium as gym
import numpy as np

# ===================== 1. ENVIRONMENT SETUP (TAXI-V3) =====================

def make_taxi_env(render_mode='ansi'):
    """
    Create Taxi-v3 environment.
    """
    env=gym.make('Taxi-v3',render_mode=render_mode)
    return env

# ===================== 2. GENERIC VALUE ITERATION FOR DISCRETE ENVS =====================

def value_iteration_discrete(env,gamma=0.99,theta=1e-8):
    """
    Value iteration for any discrete env with env.unwrapped.P.
    Returns V (state values).
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P

    V=np.zeros(nS)
    while True:
        delta=0
        for s in range(nS):
            v=V[s]
            q_sa=[]
            for a in range(nA):
                q=0
                for prob,next_state,reward,done in P[s][a]:
                    q+=prob*(reward+gamma*V[next_state])
                q_sa.append(q)
            V[s]=max(q_sa)
            delta=max(delta,abs(v-V[s]))
        if delta<theta:
            break
    return V

# Taxi-specific wrapper (just for clarity in lab)
def value_iteration_taxi(env,gamma=0.99,theta=1e-8):
    return value_iteration_discrete(env,gamma,theta)

# ===================== 3. DERIVE GREEDY POLICY FROM V =====================

def derive_greedy_policy_discrete(env,V,gamma=0.99):
    """
    Derive greedy deterministic policy from V for any discrete env.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P
    policy=np.zeros((nS,nA))

    for s in range(nS):
        q_sa=np.zeros(nA)
        for a in range(nA):
            for prob,next_state,reward,done in P[s][a]:
                q_sa[a]+=prob*(reward+gamma*V[next_state])
        best_action=int(np.argmax(q_sa))
        policy[s][best_action]=1.0
    return policy

def derive_greedy_policy_taxi(env,V,gamma=0.99):
    return derive_greedy_policy_discrete(env,V,gamma)

# ===================== 4. RANDOM POLICY INITIALISATION (TAXI) =====================

def init_random_policy_taxi(env):
    """
    Uniform random policy over all Taxi-v3 actions.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA
    print("Random Taxi policy shape:",policy.shape)
    return policy

# ===================== 5. POLICY EVALUATION (TAXI) =====================

def policy_evaluation_discrete(env,policy,discount_factor=1.0,theta=1e-9):
    """
    Policy evaluation for any discrete env with transition model P.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    V=np.zeros(nS)
    P=env.unwrapped.P

    while True:
        delta=0
        for s in range(nS):
            v=0
            for a,action_prob in enumerate(policy[s]):
                for prob,next_state,reward,done in P[s][a]:
                    v+=action_prob*prob*(reward+discount_factor*V[next_state])
            delta=max(delta,abs(V[s]-v))
            V[s]=v
        if delta<theta:
            break
    return V

def policy_evaluation_taxi(env,policy,discount_factor=0.99,theta=1e-9):
    return policy_evaluation_discrete(env,policy,discount_factor,theta)

# ===================== 6. TAXI-SPECIFIC HELPERS (DECODE & INSPECT) =====================

# Taxi actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
TAXI_ACTION_NAMES={0:'South',1:'North',2:'East',3:'West',4:'Pickup',5:'Dropoff'}

def decode_taxi_state(env,s):
    """
    Decode Taxi-v3 state index into (row,col,passenger_location,destination).
    Uses env.unwrapped.decode if available.
    """
    return env.unwrapped.decode(s)  # (taxi_row,taxi_col,pass_idx,dest_idx)

def print_taxi_state_info(env,s):
    taxi_row,taxi_col,pass_idx,dest_idx=decode_taxi_state(env,s)
    print(f"State {s}: row={taxi_row}, col={taxi_col}, passenger_idx={pass_idx}, dest_idx={dest_idx}")

def print_taxi_policy_for_states(env,policy,states):
    """
    Print greedy action (under given policy) for a list of state indices.
    """
    for s in states:
        best_a=int(np.argmax(policy[s]))
        print_taxi_state_info(env,s)
        print("  Best action:",best_a,"->",TAXI_ACTION_NAMES.get(best_a,'?'))
        print()

# Optional: simple summary of value function
def summarize_taxi_values(V,top_k=5):
    """
    Print some stats for Taxi value function:
    - min, max, mean
    - top_k states with highest value
    """
    print("V stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(np.min(V),np.max(V),np.mean(V)))
    idx_sorted=np.argsort(-V)  # descending
    print(f"Top {top_k} states by value:")
    for i in range(min(top_k,len(V))):
        s=idx_sorted[i]
        print(f"  state {s}: V={V[s]:.3f}")

# ===================== 7. QUICK DEMO WORKFLOW =====================

if __name__=="__main__":
    env=make_taxi_env()

    # --- Random policy + policy evaluation ---
    rand_policy=init_random_policy_taxi(env)
    V_rand=policy_evaluation_taxi(env,rand_policy,discount_factor=0.99,theta=1e-9)
    print("Random policy V stats:")
    summarize_taxi_values(V_rand,top_k=5)

    # --- Optimal value via value iteration + greedy policy ---
    V_opt=value_iteration_taxi(env,gamma=0.99,theta=1e-8)
    print("\nOptimal V stats (value iteration):")
    summarize_taxi_values(V_opt,top_k=5)

    opt_policy=derive_greedy_policy_taxi(env,V_opt,gamma=0.99)

    # Inspect policy for a few arbitrary states:
    interesting_states=[0,100,250,499]
    print("\nGreedy policy actions for some states:")
    print_taxi_policy_for_states(env,opt_policy,interesting_states)

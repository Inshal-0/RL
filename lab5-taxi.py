import gymnasium as gym
import numpy as np

# ===================== 1. ENVIRONMENT SETUP (TAXI-V3) =====================

def make_taxi_env(render_mode='ansi'):
    """
    Create Taxi-v3 environment.
    """
    env=gym.make('Taxi-v3',render_mode=render_mode)
    return env

# ===================== 2. Q-FROM-V FOR TAXI =====================

def q_from_v_taxi(env,V,s,gamma=1.0):
    """
    Compute Q(s,a) for all actions a in Taxi-v3, given state values V.
    Uses env.unwrapped.P transition model.
    """
    P=env.unwrapped.P
    nA=env.action_space.n
    q=np.zeros(nA)
    for a in range(nA):
        for prob,next_state,reward,done in P[s][a]:
            q[a]+=prob*(reward+gamma*V[next_state])
    return q

# ===================== 3. POLICY IMPROVEMENT (GREEDY) =====================

def policy_improvement_taxi(env,V,discount_factor=1.0):
    """
    Greedy policy improvement for Taxi-v3:
      π'(s)=argmax_a Q(s,a)
    Returns deterministic one-hot policy [nS,nA].
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.zeros((nS,nA))
    for s in range(nS):
        Q=q_from_v_taxi(env,V,s,discount_factor)
        best_action=int(np.argmax(Q))
        policy[s]=np.eye(nA)[best_action]
    return policy

# ===================== 4. POLICY EVALUATION (FOR TAXI) =====================

def policy_evaluation_taxi(env,policy,discount_factor=1.0,theta=1e-9):
    """
    Policy evaluation for Taxi-v3 using Bellman expectation equation.
    Given policy π, compute V^π.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P
    V=np.zeros(nS)

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

# ===================== 5. FULL POLICY ITERATION (TAXI) =====================

def policy_iteration_taxi(env,discount_factor=1.0,theta=1e-9,max_iterations=1000,verbose=False):
    """
    Policy Iteration for Taxi-v3:
      repeat:
        1) Policy Evaluation
        2) Policy Improvement
      until policy stable.
    Returns (policy,V,num_iterations,stable_flag).
    """
    nS=env.observation_space.n
    nA=env.action_space.n

    # start with uniform random policy
    policy=np.ones((nS,nA))/nA

    for i in range(max_iterations):
        V=policy_evaluation_taxi(env,policy,discount_factor,theta)
        new_policy=policy_improvement_taxi(env,V,discount_factor)
        stable=np.array_equal(new_policy,policy)
        if verbose:
            print("Iteration",i+1,"stable:",stable)
        policy=new_policy
        if stable:
            return policy,V,i+1,True

    return policy,V,max_iterations,False

# ===================== 6. HELPERS TO INSPECT TAXI POLICY =====================

TAXI_ACTION_NAMES={0:'South',1:'North',2:'East',3:'West',4:'Pickup',5:'Dropoff'}

def decode_taxi_state(env,s):
    """
    Decode Taxi-v3 state index into (taxi_row,taxi_col,passenger_idx,dest_idx).
    """
    return env.unwrapped.decode(s)

def print_taxi_state_info(env,s):
    taxi_row,taxi_col,pass_idx,dest_idx=decode_taxi_state(env,s)
    print(f"State {s}: row={taxi_row}, col={taxi_col}, passenger_idx={pass_idx}, dest_idx={dest_idx}")

def print_taxi_policy_for_states(env,policy,states):
    """
    Print greedy action chosen by policy for given list of state indices.
    """
    for s in states:
        best_a=int(np.argmax(policy[s]))
        print_taxi_state_info(env,s)
        print("  Best action:",best_a,"->",TAXI_ACTION_NAMES.get(best_a,'?'))
        print()

def summarize_taxi_values(V,top_k=5):
    """
    Print summary stats of V and top_k highest-value states.
    """
    print("V stats: min={:.3f}, max={:.3f}, mean={:.3f}"
          .format(np.min(V),np.max(V),np.mean(V)))
    idx_sorted=np.argsort(-V)
    print(f"Top {top_k} states by value:")
    for i in range(min(top_k,len(V))):
        s=idx_sorted[i]
        print(f"  state {s}: V={V[s]:.3f}")

# ===================== 7. QUICK DEMO =====================

if __name__=="__main__":
    env=make_taxi_env()

    # Example: random V -> greedy policy improvement
    V=np.random.rand(env.observation_space.n)
    greedy_policy=policy_improvement_taxi(env,V,discount_factor=0.99)
    print("Greedy policy from random V created with shape:",greedy_policy.shape)

    # Policy Iteration to get optimal policy
    opt_policy,opt_V,iters,stable=policy_iteration_taxi(env,discount_factor=0.99,theta=1e-9,verbose=True)
    print("Taxi policy iteration stable:",stable,"after",iters,"iterations")
    summarize_taxi_values(opt_V,top_k=5)

    # Inspect optimal actions for some example states
    example_states=[0,100,250,499]
    print_taxi_policy_for_states(env,opt_policy,example_states)

# Policy improvement from a given V:

env=make_taxi_env()
V=np.random.rand(env.observation_space.n)
policy=policy_improvement_taxi(env,V,discount_factor=0.99)


# Full policy iteration (Task 4 type):

opt_policy,opt_V,iters,stable=policy_iteration_taxi(env,discount_factor=0.99,theta=1e-9,verbose=True)



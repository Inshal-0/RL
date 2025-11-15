import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP =====================

def make_taxi_env(render_mode='ansi'):
    """
    Create Taxi-v3 environment.
    """
    env=gym.make('Taxi-v3',render_mode=render_mode)
    return env

def describe_env_taxi(env):
    """
    Print basic info about Taxi-v3 MDP.
    """
    print("Observation space (nS):",env.observation_space.n)
    print("Action space (nA):",env.action_space.n)
    print("Actions: 0=South,1=North,2=East,3=West,4=Pickup,5=Dropoff")
    print("Rewards (default Gym):")
    print("  -1 per timestep, +20 on successful dropoff, -10 for illegal pickup/dropoff.")
    print("Taxi is a finite MDP: finite S,A,known P(s'|s,a),reward function,R,discount gamma.")

# ===================== 2. RANDOM POLICY =====================

def init_random_policy_taxi(env):
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA
    return policy

# ===================== 3. DP POLICY EVALUATION (TAXI) =====================

def policy_evaluation_dp_taxi(env,policy,discount_factor=1.0,theta=1e-9,
                              return_history=False,max_iters=1000):
    """
    Iterative policy evaluation for Taxi-v3 using Bellman expectation eq.
    If return_history=True, return [V_0,V_1,...] across sweeps.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P
    V=np.zeros(nS)
    history=[]

    for it in range(max_iters):
        delta=0
        for s in range(nS):
            v=0
            for a,action_prob in enumerate(policy[s]):
                for prob,next_state,reward,done in P[s][a]:
                    v+=action_prob*prob*(reward+discount_factor*V[next_state])
            delta=max(delta,abs(V[s]-v))
            V[s]=v
        if return_history:
            history.append(V.copy())
        if delta<theta:
            break
    if return_history:
        return V,history
    return V

def plot_value_convergence_taxi(history,state_idx=0):
    """
    Plot V_t(s) across iterations for some chosen state index.
    """
    vals=[V[state_idx] for V in history]
    plt.figure()
    plt.plot(vals)
    plt.xlabel("Iteration")
    plt.ylabel(f"V(s={state_idx})")
    plt.title("Taxi: Value Convergence for State "+str(state_idx))
    plt.grid(True)
    plt.show()

# ===================== 4. Q-FROM-V + POLICY IMPROVEMENT =====================

def q_from_v_taxi(env,V,s,gamma=1.0):
    """
    Compute Q(s,a) for Taxi-v3 given V.
    """
    P=env.unwrapped.P
    nA=env.action_space.n
    q=np.zeros(nA)
    for a in range(nA):
        for prob,next_state,reward,done in P[s][a]:
            q[a]+=prob*(reward+gamma*V[next_state])
    return q

def policy_improvement_taxi(env,V,discount_factor=1.0):
    """
    Greedy policy improvement:
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

# ===================== 5. FULL POLICY ITERATION (TAXI) =====================

def policy_iteration_taxi(env,discount_factor=1.0,theta=1e-9,
                          max_iterations=1000,verbose=False):
    """
    Policy Iteration:
      repeat:
        V <- policy evaluation
        π <- policy improvement
      until policy stable.
    Returns: policy,V,num_iterations,stable_flag
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA

    for i in range(max_iterations):
        V=policy_evaluation_dp_taxi(env,policy,discount_factor,theta)
        new_policy=policy_improvement_taxi(env,V,discount_factor)
        stable=np.array_equal(new_policy,policy)
        if verbose:
            print("Policy iteration step",i+1,"stable:",stable)
        policy=new_policy
        if stable:
            return policy,V,i+1,True
    return policy,V,max_iterations,False

# ===================== 6. MONTE CARLO EVALUATION (EPISODE SAMPLING) =====================

def run_episode_taxi(env,policy,discount_factor=1.0,max_steps=200):
    """
    Run a single episode under given policy; return discounted return G and steps.
    """
    state,_=env.reset()
    G=0.0
    gamma_pow=1.0
    for t in range(max_steps):
        action=np.random.choice(env.action_space.n,p=policy[state])
        next_state,reward,terminated,truncated,info=env.step(action)
        G+=gamma_pow*reward
        gamma_pow*=discount_factor
        state=next_state
        if terminated or truncated:
            return G,t+1
    return G,max_steps

def mc_evaluate_policy_taxi(env,policy,discount_factor=1.0,num_episodes=1000):
    """
    Monte Carlo estimate of expected return from start state under given policy.
    """
    returns=[]
    for _ in range(num_episodes):
        G,_=run_episode_taxi(env,policy,discount_factor)
        returns.append(G)
    return np.mean(returns),np.array(returns)

# ===================== 7. EXPERIMENT: COMPARE DIFFERENT GAMMA VALUES =====================

def experiment_compare_gamma_taxi(gamma1=0.9,gamma2=0.99,
                                  theta=1e-9,num_eval_episodes=500):
    """
    Compare policy iteration + MC performance for two different discount factors.
    Returns dict with stats for each gamma.
    """
    env1=make_taxi_env()
    pol1,V1,it1,stab1=policy_iteration_taxi(env1,discount_factor=gamma1,theta=theta,verbose=False)
    avg_ret1,_=mc_evaluate_policy_taxi(env1,pol1,discount_factor=gamma1,num_episodes=num_eval_episodes)

    env2=make_taxi_env()
    pol2,V2,it2,stab2=policy_iteration_taxi(env2,discount_factor=gamma2,theta=theta,verbose=False)
    avg_ret2,_=mc_evaluate_policy_taxi(env2,pol2,discount_factor=gamma2,num_episodes=num_eval_episodes)

    stats={
        "gamma1":{"gamma":gamma1,"iterations":it1,"stable":stab1,"avg_return":avg_ret1,
                  "V":V1,"policy":pol1,"env":env1},
        "gamma2":{"gamma":gamma2,"iterations":it2,"stable":stab2,"avg_return":avg_ret2,
                  "V":V2,"policy":pol2,"env":env2}
    }
    return stats

# ===================== 8. OPTIONAL HELPERS TO INSPECT STATES/POLICY =====================

TAXI_ACTION_NAMES={0:'South',1:'North',2:'East',3:'West',4:'Pickup',5:'Dropoff'}

def decode_taxi_state(env,s):
    return env.unwrapped.decode(s)  # (row,col,pass_idx,dest_idx)

def print_taxi_state_info(env,s):
    row,col,pass_idx,dest_idx=decode_taxi_state(env,s)
    print(f"State {s}: row={row}, col={col}, passenger_idx={pass_idx}, dest_idx={dest_idx}")

def print_taxi_policy_for_states(env,policy,states):
    for s in states:
        best_a=int(np.argmax(policy[s]))
        print_taxi_state_info(env,s)
        print("  Best action:",best_a,"->",TAXI_ACTION_NAMES.get(best_a,'?'))
        print()

def summarize_taxi_values(V,top_k=5):
    print("V stats: min={:.3f}, max={:.3f}, mean={:.3f}"
          .format(np.min(V),np.max(V),np.mean(V)))
    idx_sorted=np.argsort(-V)
    print(f"Top {top_k} states by value:")
    for i in range(min(top_k,len(V))):
        s=idx_sorted[i]
        print(f"  state {s}: V={V[s]:.3f}")

# ===================== 9. QUICK DEMO PIPELINE =====================

if __name__=="__main__":
    # 1) Env setup
    env=make_taxi_env()
    describe_env_taxi(env)

    # 2) Start with random policy and do DP policy evaluation
    rand_policy=init_random_policy_taxi(env)
    V_rand,history=policy_evaluation_dp_taxi(env,rand_policy,discount_factor=0.99,
                                             theta=1e-9,return_history=True)
    print("Random policy V (first 10 states):",V_rand[:10])
    plot_value_convergence_taxi(history,state_idx=0)

    # 3) Policy iteration to get optimal policy
    opt_policy,opt_V,iters,stable=policy_iteration_taxi(env,discount_factor=0.99,theta=1e-9,verbose=True)
    print("Taxi policy iteration stable:",stable,"after",iters,"iterations")
    summarize_taxi_values(opt_V,top_k=5)
    print_taxi_policy_for_states(env,opt_policy,[0,50,100,250,499])

    # 4) Experimental analysis: compare gamma values
    stats=experiment_compare_gamma_taxi(gamma1=0.9,gamma2=0.99,theta=1e-9,num_eval_episodes=500)
    print("\nGamma={:.2f}: iterations={}, avg_return={:.3f}"
          .format(stats["gamma1"]["gamma"],stats["gamma1"]["iterations"],stats["gamma1"]["avg_return"]))
    print("Gamma={:.2f}: iterations={}, avg_return={:.3f}"
          .format(stats["gamma2"]["gamma"],stats["gamma2"]["gamma"],stats["gamma2"]["avg_return"]))

    # 5) MC vs DP for optimal policy (gamma2)
    env_opt=stats["gamma2"]["env"]
    pol_opt=stats["gamma2"]["policy"]
    V_opt=stats["gamma2"]["V"]
    mc_mean,_=mc_evaluate_policy_taxi(env_opt,pol_opt,discount_factor=stats["gamma2"]["gamma"],num_episodes=2000)
    print("\nMC estimate of start-state value (optimal policy):",mc_mean)
    print("DP value at start state:",V_opt[0])

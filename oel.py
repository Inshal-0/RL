import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP =====================

def make_frozenlake_env(size=4,is_slippery=True,render_mode='ansi'):
    """
    Create FrozenLake-v1 environment (4x4 or 8x8).
    size: 4 or 8
    """
    map_name='4x4' if size==4 else '8x8'
    env=gym.make('FrozenLake-v1',map_name=map_name,is_slippery=is_slippery,render_mode=render_mode)
    return env

def describe_env(env):
    """
    Print basic info: states, actions, rewards (informally).
    """
    print("Observation space (nS):",env.observation_space.n)
    print("Action space (nA):",env.action_space.n)
    print("Actions: 0=Left,1=Down,2=Right,3=Up")
    print("Rewards: 0 for non-terminal steps, 1 at goal, 0 at holes.")
    print("MDP properties: finite S,A,known P(s'|s,a),reward function,R,discount gamma.")

# ===================== 2. RANDOM POLICY =====================

def init_random_policy(env):
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA
    return policy

# ===================== 3. DYNAMIC PROGRAMMING POLICY EVALUATION =====================

def policy_evaluation_dp(env,policy,discount_factor=1.0,theta=1e-9,return_history=False,max_iters=1000):
    """
    Iterative policy evaluation using Bellman expectation equation.
    If return_history=True, also return list of V after each sweep.
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

def plot_value_convergence(history,state_idx=0):
    """
    Plot V_t(s) across iterations for a selected state index.
    """
    vals=[V[state_idx] for V in history]
    plt.figure()
    plt.plot(vals)
    plt.xlabel("Iteration")
    plt.ylabel(f"V(s={state_idx})")
    plt.title("Value Convergence for State "+str(state_idx))
    plt.grid(True)
    plt.show()

# ===================== 4. Q-FROM-V + POLICY IMPROVEMENT =====================

def q_from_v(env,V,s,gamma=1.0):
    P=env.unwrapped.P
    nA=env.action_space.n
    q=np.zeros(nA)
    for a in range(nA):
        for prob,next_state,reward,done in P[s][a]:
            q[a]+=prob*(reward+gamma*V[next_state])
    return q

def policy_improvement(env,V,discount_factor=1.0):
    """
    Greedy policy improvement:
      π'(s)=argmax_a Q(s,a)
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.zeros((nS,nA))
    for s in range(nS):
        Q=q_from_v(env,V,s,discount_factor)
        best_action=int(np.argmax(Q))
        policy[s]=np.eye(nA)[best_action]
    return policy

# ===================== 5. FULL POLICY ITERATION =====================

def policy_iteration(env,discount_factor=1.0,theta=1e-9,max_iterations=1000,verbose=False):
    """
    Alternate policy evaluation and improvement until policy stable.
    Returns: policy,V,num_iterations,stable_flag
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA

    for i in range(max_iterations):
        V=policy_evaluation_dp(env,policy,discount_factor,theta)
        new_policy=policy_improvement(env,V,discount_factor)
        stable=np.array_equal(new_policy,policy)
        if verbose:
            print("Policy iteration step",i+1,"stable:",stable)
        policy=new_policy
        if stable:
            return policy,V,i+1,True
    return policy,V,max_iterations,False

# ===================== 6. VISUALIZATION (VALUE + POLICY) =====================

def plot_frozenlake(V,policy,env,draw_vals=True):
    """
    Heatmap + arrows visualization.
    draw_vals=True -> show V(s), False -> show arrows.
    """
    nrow=env.unwrapped.nrow
    ncol=env.unwrapped.ncol
    arrow_symbols={0:'←',1:'↓',2:'→',3:'↑'}
    grid=np.reshape(V,(nrow,ncol))
    plt.figure(figsize=(4,4))
    plt.imshow(grid,cmap='cool',interpolation='none')

    for s in range(nrow*ncol):
        r,c=divmod(s,ncol)
        best_action=int(np.argmax(policy[s]))
        if draw_vals:
            plt.text(c,r,f"{V[s]:.2f}",ha='center',va='center',color='white',fontsize=8)
        else:
            plt.text(c,r,arrow_symbols[best_action],ha='center',va='center',color='white',fontsize=12)

    plt.title("Value Function" if draw_vals else "Policy")
    plt.axis('off')
    plt.show()

# ===================== 7. MONTE CARLO POLICY EVALUATION (SAMPLING) =====================

def run_episode(env,policy,discount_factor=1.0,max_steps=100):
    """
    Run one episode following 'policy' and return discounted return G and steps.
    Episode starts from default start state.
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

def mc_evaluate_policy(env,policy,discount_factor=1.0,num_episodes=1000):
    """
    Monte Carlo estimate of expected return from start state
    under given policy.
    """
    returns=[]
    for _ in range(num_episodes):
        G,_=run_episode(env,policy,discount_factor)
        returns.append(G)
    return np.mean(returns),np.array(returns)

# ===================== 8. EXPERIMENT: COMPARE ENVIRONMENT SETTINGS =====================

def experiment_compare_slip(size=4,gamma=0.99,theta=1e-9,num_eval_episodes=500):
    """
    Compare policy iteration results between:
      - slippery=True
      - slippery=False
    Returns dict with stats.
    """
    # slippery env
    env_slip=make_frozenlake_env(size=size,is_slippery=True)
    pol_s,V_s,it_s,stab_s=policy_iteration(env_slip,discount_factor=gamma,theta=theta,verbose=False)
    avg_return_s,_=mc_evaluate_policy(env_slip,pol_s,discount_factor=gamma,num_episodes=num_eval_episodes)

    # non-slippery env
    env_det=make_frozenlake_env(size=size,is_slippery=False)
    pol_d,V_d,it_d,stab_d=policy_iteration(env_det,discount_factor=gamma,theta=theta,verbose=False)
    avg_return_d,_=mc_evaluate_policy(env_det,pol_d,discount_factor=gamma,num_episodes=num_eval_episodes)

    stats={
        "slippery":{"iterations":it_s,"stable":stab_s,"avg_return":avg_return_s,"V":V_s,"policy":pol_s,"env":env_slip},
        "deterministic":{"iterations":it_d,"stable":stab_d,"avg_return":avg_return_d,"V":V_d,"policy":pol_d,"env":env_det}
    }
    return stats

# ===================== 9. QUICK DEMO PIPELINE =====================

if __name__=="__main__":
    # 1) Environment setup
    env=make_frozenlake_env(size=4,is_slippery=True)
    describe_env(env)

    # 2) Start with random policy and DP policy evaluation
    rand_policy=init_random_policy(env)
    V_rand,history=policy_evaluation_dp(env,rand_policy,discount_factor=0.99,theta=1e-9,return_history=True)
    print("Random policy V (first few):",V_rand[:5])
    plot_value_convergence(history,state_idx=0)

    # 3) Policy improvement + policy iteration to optimal
    opt_policy,opt_V,iters,stable=policy_iteration(env,discount_factor=0.99,theta=1e-9,verbose=True)
    print("Policy iteration stable:",stable,"after",iters,"iterations")
    plot_frozenlake(opt_V,opt_policy,env,draw_vals=True)
    plot_frozenlake(opt_V,opt_policy,env,draw_vals=False)

    # 4) Experimental analysis: compare slippery vs deterministic
    stats=experiment_compare_slip(size=4,gamma=0.99,theta=1e-9,num_eval_episodes=500)
    print("\nSlippery env: iterations={}, avg_return={:.3f}"
          .format(stats["slippery"]["iterations"],stats["slippery"]["avg_return"]))
    print("Deterministic env: iterations={}, avg_return={:.3f}"
          .format(stats["deterministic"]["iterations"],stats["deterministic"]["avg_return"]))

    # 5) Monte Carlo vs DP example for reflection
    mc_mean,_=mc_evaluate_policy(env,opt_policy,discount_factor=0.99,num_episodes=2000)
    print("\nMC estimate of start-state value (optimal policy):",mc_mean)
    print("DP value at start state:",opt_V[0])

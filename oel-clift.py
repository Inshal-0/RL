import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP (CLIFFWALKING-V1) =====================

def make_cliff_env(render_mode='ansi'):
    """
    Create CliffWalking-v1 environment (4x12 grid).
    """
    env=gym.make('CliffWalking-v1',render_mode=render_mode)
    return env

def describe_env_cliff(env):
    """
    Print basic info about CliffWalking MDP.
    """
    print("Observation space (nS):",env.observation_space.n)
    print("Action space (nA):",env.action_space.n)
    print("Actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT")
    print("Rewards: -1 per step, -100 for stepping into cliff, 0 at goal.")
    print("Grid: 4x12, bottom row has cliff cells between start and goal.")
    print("Finite MDP with known P(s'|s,a), rewards R, and discount gamma.")

# ===================== 2. RANDOM POLICY =====================

def init_random_policy_cliff(env):
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA
    return policy

# ===================== 3. DP POLICY EVALUATION =====================

def policy_evaluation_dp_cliff(env,policy,discount_factor=1.0,theta=1e-9,
                               return_history=False,max_iters=1000):
    """
    Iterative policy evaluation (Bellman expectation) for CliffWalking.
    If return_history=True, returns list of V after each sweep.
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

def plot_value_convergence_cliff(history,state_idx=0):
    """
    Plot V_t(s) across iterations for a chosen state index.
    """
    vals=[V[state_idx] for V in history]
    plt.figure()
    plt.plot(vals)
    plt.xlabel("Iteration")
    plt.ylabel(f"V(s={state_idx})")
    plt.title("CliffWalking-v1: Value Convergence for State "+str(state_idx))
    plt.grid(True)
    plt.show()

# ===================== 4. Q-FROM-V + POLICY IMPROVEMENT =====================

def q_from_v_cliff(env,V,s,gamma=1.0):
    """
    Compute Q(s,a) for CliffWalking given V.
    """
    P=env.unwrapped.P
    nA=env.action_space.n
    q=np.zeros(nA)
    for a in range(nA):
        for prob,next_state,reward,done in P[s][a]:
            q[a]+=prob*(reward+gamma*V[next_state])
    return q

def policy_improvement_cliff(env,V,discount_factor=1.0):
    """
    Greedy policy improvement:
      π'(s)=argmax_a Q(s,a)
    Returns deterministic one-hot policy [nS,nA].
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.zeros((nS,nA))
    for s in range(nS):
        Q=q_from_v_cliff(env,V,s,discount_factor)
        best_action=int(np.argmax(Q))
        policy[s]=np.eye(nA)[best_action]
    return policy

# ===================== 5. FULL POLICY ITERATION =====================

def policy_iteration_cliff(env,discount_factor=1.0,theta=1e-9,
                           max_iterations=1000,verbose=False):
    """
    Policy Iteration for CliffWalking-v1:
      V <- policy evaluation
      π <- policy improvement
    until policy is stable.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA

    for i in range(max_iterations):
        V=policy_evaluation_dp_cliff(env,policy,discount_factor,theta)
        new_policy=policy_improvement_cliff(env,V,discount_factor)
        stable=np.array_equal(new_policy,policy)
        if verbose:
            print("Policy iteration step",i+1,"stable:",stable)
        policy=new_policy
        if stable:
            return policy,V,i+1,True
    return policy,V,max_iterations,False

# ===================== 6. MONTE CARLO EVALUATION =====================

def run_episode_cliff(env,policy,discount_factor=1.0,max_steps=500):
    """
    Run one episode under given policy from the default start state.
    Returns discounted return G and number of steps.
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

def mc_evaluate_policy_cliff(env,policy,discount_factor=1.0,num_episodes=1000):
    """
    Monte Carlo estimate of expected return from start under given policy.
    """
    returns=[]
    for _ in range(num_episodes):
        G,_=run_episode_cliff(env,policy,discount_factor)
        returns.append(G)
    return np.mean(returns),np.array(returns)

# ===================== 7. EXPERIMENT: COMPARE DIFFERENT GAMMA VALUES =====================

def experiment_compare_gamma_cliff(gamma1=0.9,gamma2=0.99,
                                   theta=1e-9,num_eval_episodes=500):
    """
    Compare policy iteration + MC performance for two discount factors
    in CliffWalking-v1.
    """
    env1=make_cliff_env()
    pol1,V1,it1,stab1=policy_iteration_cliff(env1,discount_factor=gamma1,theta=theta,verbose=False)
    avg_ret1,_=mc_evaluate_policy_cliff(env1,pol1,discount_factor=gamma1,num_episodes=num_eval_episodes)

    env2=make_cliff_env()
    pol2,V2,it2,stab2=policy_iteration_cliff(env2,discount_factor=gamma2,theta=theta,verbose=False)
    avg_ret2,_=mc_evaluate_policy_cliff(env2,pol2,discount_factor=gamma2,num_episodes=num_eval_episodes)

    stats={
        "gamma1":{"gamma":gamma1,"iterations":it1,"stable":stab1,"avg_return":avg_ret1,
                  "V":V1,"policy":pol1,"env":env1},
        "gamma2":{"gamma":gamma2,"iterations":it2,"stable":stab2,"avg_return":avg_ret2,
                  "V":V2,"policy":pol2,"env":env2}
    }
    return stats

# ===================== 8. VISUALIZATION: VALUE + POLICY GRID =====================

# Cliff actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
CLIFF_ARROW_DICT={0:'↑',1:'→',2:'↓',3:'←'}

def plot_cliff(V,policy,env,draw_vals=True):
    """
    Visualize CliffWalking-v1 grid:
      - S (start), G (goal), C (cliff), . (safe)
      - V(s) (heatmap) or arrows for greedy actions.
    """
    nrow,ncol=env.unwrapped.shape  # typically (4,12)
    V_sq=V.reshape((nrow,ncol))

    plt.figure(figsize=(5,2.5))
    plt.imshow(V_sq,cmap='cool',alpha=0.7)
    ax=plt.gca()

    for x in range(ncol+1):
        ax.axvline(x-0.5,lw=0.5,color='black')
    for y in range(nrow+1):
        ax.axhline(y-0.5,lw=0.5,color='black')

    start=(nrow-1,0)
    goal=(nrow-1,ncol-1)
    cliff=[(nrow-1,c) for c in range(1,ncol-1)]

    for r in range(nrow):
        for c in range(ncol):
            s=r*ncol+c
            val=V[s]

            if (r,c)==start:
                tile='S'
                color='blue'
            elif (r,c)==goal:
                tile='G'
                color='green'
            elif (r,c) in cliff:
                tile='C'
                color='red'
            else:
                tile='.'
                color='black'

            plt.text(c,r,tile,ha='center',va='center',color=color,
                     fontsize=10,fontweight='bold')

            if draw_vals and tile!='C':
                plt.text(c,r+0.35,f"{val:.1f}",ha='center',va='center',
                         color='black',fontsize=6)
            elif not draw_vals and policy is not None and tile!='C':
                best_action=int(np.argmax(policy[s]))
                arrow=CLIFF_ARROW_DICT.get(best_action,'')
                plt.text(c,r-0.25,arrow,ha='center',va='center',
                         color='purple',fontsize=10)

    plt.title("CliffWalking-v1: "+("Value Function" if draw_vals else "Policy"))
    plt.axis('off')
    plt.show()

# ===================== 9. QUICK DEMO PIPELINE =====================

if __name__=="__main__":
    env=make_cliff_env()
    describe_env_cliff(env)

    # Random policy + evaluation
    rand_policy=init_random_policy_cliff(env)
    V_rand,history=policy_evaluation_dp_cliff(env,rand_policy,discount_factor=0.99,
                                              theta=1e-9,return_history=True)
    print("Random policy V (first 10 states):",V_rand[:10])
    plot_value_convergence_cliff(history,state_idx=0)

    # Policy iteration to optimal policy
    opt_policy,opt_V,iters,stable=policy_iteration_cliff(env,discount_factor=0.99,theta=1e-9,verbose=True)
    print("Cliff-v1 policy iteration stable:",stable,"after",iters,"iterations")
    plot_cliff(opt_V,opt_policy,env,draw_vals=True)
    plot_cliff(opt_V,opt_policy,env,draw_vals=False)

    # Gamma comparison
    stats=experiment_compare_gamma_cliff(gamma1=0.9,gamma2=0.99,theta=1e-9,num_eval_episodes=500)
    print("\nGamma={:.2f}: iterations={}, avg_return={:.3f}"
          .format(stats["gamma1"]["gamma"],stats["gamma1"]["iterations"],stats["gamma1"]["avg_return"]))
    print("Gamma={:.2f}: iterations={}, avg_return={:.3f}"
          .format(stats["gamma2"]["gamma"],stats["gamma2"]["iterations"],stats["gamma2"]["avg_return"]))

    env_opt=stats["gamma2"]["env"]
    pol_opt=stats["gamma2"]["policy"]
    V_opt=stats["gamma2"]["V"]
    mc_mean,_=mc_evaluate_policy_cliff(env_opt,pol_opt,discount_factor=stats["gamma2"]["gamma"],num_episodes=2000)
    print("\nMC estimate of start-state value (optimal policy):",mc_mean)
    print("DP value at start state:",V_opt[0])

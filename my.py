import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. GENERIC ENVIRONMENT INITIALIZATION
#    (Task 1: "initialize an environment, could be taxi, cliff, frozenlake")
# ============================================================

def make_env(env_type="frozenlake",size=4,is_slippery=True,render_mode='ansi'):
    """
    env_type: "frozenlake", "taxi", "cliff"
    size: 4 or 8 for FrozenLake
    """
    if env_type.lower()=="frozenlake":
        map_name='4x4' if size==4 else '8x8'
        env=gym.make('FrozenLake-v1',map_name=map_name,
                     is_slippery=is_slippery,render_mode=render_mode)
    elif env_type.lower()=="taxi":
        env=gym.make('Taxi-v3',render_mode=render_mode)
    elif env_type.lower()=="cliff":
        env=gym.make('CliffWalking-v1',render_mode=render_mode)
    else:
        raise ValueError("Unknown env_type. Use 'frozenlake','taxi','cliff'.")
    return env

def init_random_policy(env):
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA
    return policy

# ============================================================
# 2. ITERATIVE POLICY EVALUATION (DP) + PRINT PER ITERATION
#    (Task 2: evaluate random policy, print V each iter, plot final V)
# ============================================================

def policy_evaluation_dp(env,policy,discount_factor=1.0,theta=1e-9,
                         max_iters=1000,print_each=True,return_history=False):
    """
    Iterative policy evaluation with optional printing and history.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P
    V=np.zeros(nS)
    history=[]

    for it in range(max_iters):
        delta=0
        V_new=np.zeros_like(V)
        for s in range(nS):
            v=0
            for a,action_prob in enumerate(policy[s]):
                for prob,next_state,reward,done in P[s][a]:
                    v+=action_prob*prob*(reward+discount_factor*V[next_state])
            V_new[s]=v
            delta=max(delta,abs(V[s]-v))
        V=V_new
        if return_history:
            history.append(V.copy())
        if print_each:
            print(f"Iteration {it+1}, V:",V)
        if delta<theta:
            break
    if return_history:
        return V,history
    return V

def plot_values(env,V,title="State Values"):
    """
    Plot state values.
    - For grid envs (FrozenLake/Cliff) -> heatmap.
    - For Taxi -> bar plot of first N states.
    """
    if hasattr(env.unwrapped,'nrow') and hasattr(env.unwrapped,'ncol'):
        nrow=env.unwrapped.nrow
        ncol=env.unwrapped.ncol
        grid=V.reshape((nrow,ncol))
        plt.figure(figsize=(4,4))
        plt.imshow(grid,cmap='cool',interpolation='none')
        for s in range(nrow*ncol):
            r,c=divmod(s,ncol)
            plt.text(c,r,f"{V[s]:.2f}",ha='center',va='center',
                     color='white',fontsize=8)
        plt.title(title)
        plt.axis('off')
        plt.show()
    elif hasattr(env.unwrapped,'shape'):  # CliffWalking-v1 also has shape
        nrow,ncol=env.unwrapped.shape
        grid=V.reshape((nrow,ncol))
        plt.figure(figsize=(5,2.5))
        plt.imshow(grid,cmap='cool',interpolation='none')
        for s in range(nrow*ncol):
            r,c=divmod(s,ncol)
            plt.text(c,r,f"{V[s]:.1f}",ha='center',va='center',
                     color='white',fontsize=6)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        # Taxi or other non-grid: bar plot of first 50 states
        N=min(len(V),50)
        plt.figure(figsize=(8,3))
        plt.bar(np.arange(N),V[:N])
        plt.xlabel("State index")
        plt.ylabel("V(s)")
        plt.title(title+" (first "+str(N)+" states)")
        plt.show()

# ============================================================
# 3. Q-FROM-V & POLICY IMPROVEMENT (GREEDY)
#    (Used in policy iteration & value iteration)
# ============================================================

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

# ============================================================
# 4. POLICY ITERATION (EVAL + IMPROVE UNTIL OPTIMAL)
#    (Task 3: find optimal policy + optimal V)
# ============================================================

def policy_iteration(env,discount_factor=1.0,theta=1e-9,
                     max_iterations=1000,verbose=True):
    """
    Policy Iteration:
      repeat:
        - policy evaluation
        - policy improvement
      until policy stable.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA

    for i in range(max_iterations):
        V=policy_evaluation_dp(env,policy,discount_factor,theta,
                               max_iters=1000,print_each=False)
        new_policy=policy_improvement(env,V,discount_factor)
        stable=np.array_equal(new_policy,policy)
        if verbose:
            print("Policy iteration step",i+1,"stable:",stable)
        policy=new_policy
        if stable:
            return policy,V,i+1,True
    return policy,V,max_iterations,False

# ============================================================
# 5. VALUE ITERATION (STARTING FROM RANDOM / ZERO V)
#    (Task 4: value iteration, then plot values & policy)
# ============================================================

def value_iteration(env,gamma=1.0,theta=1e-9,max_iterations=1000,verbose=True):
    """
    Standard value iteration (doesn't actually need a policy as input).
    Returns optimal V and greedy policy wrt V.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P
    V=np.zeros(nS)

    for it in range(max_iterations):
        delta=0
        V_new=np.zeros_like(V)
        for s in range(nS):
            q_sa=[]
            for a in range(nA):
                q=0
                for prob,next_state,reward,done in P[s][a]:
                    q+=prob*(reward+gamma*V[next_state])
                q_sa.append(q)
            V_new[s]=max(q_sa)
            delta=max(delta,abs(V[s]-V_new[s]))
        V=V_new
        if verbose:
            print(f"Value iteration {it+1}, max change={delta:.6f}")
        if delta<theta:
            break

    # greedy policy from V
    policy_opt=policy_improvement(env,V,discount_factor=gamma)
    return V,policy_opt

# ============================================================
# 6. POLICY PLOTTING FOR GRID ENVS (FrozenLake / Cliff)
#    (Used for Tasks 3 & 4 when they say "plot values and policy")
# ============================================================

def plot_grid_policy(env,V,policy,title="Policy + Values"):
    """
    For grid envs only (FrozenLake or CliffWalking).
    Shows arrows + (optional) values.
    """
    # FrozenLake: nrow,ncol properties
    if hasattr(env.unwrapped,'nrow') and hasattr(env.unwrapped,'ncol'):
        nrow=env.unwrapped.nrow
        ncol=env.unwrapped.ncol
    else:
        # CliffWalking has shape
        nrow,ncol=env.unwrapped.shape

    arrow_symbols={}
    if env.spec.id.startswith("FrozenLake"):
        arrow_symbols={0:'←',1:'↓',2:'→',3:'↑'}
    else:
        # Cliff: 0=UP,1=RIGHT,2=DOWN,3=LEFT
        arrow_symbols={0:'↑',1:'→',2:'↓',3:'←'}

    grid=V.reshape((nrow,ncol))
    plt.figure(figsize=(4,4))
    plt.imshow(grid,cmap='cool',interpolation='none')

    for s in range(nrow*ncol):
        r,c=divmod(s,ncol)
        best_action=int(np.argmax(policy[s]))
        plt.text(c,r,f"{V[s]:.2f}",ha='center',va='center',
                 color='white',fontsize=7)
        plt.text(c,r-0.3,arrow_symbols[best_action],ha='center',va='center',
                 color='yellow',fontsize=10)

    plt.title(title)
    plt.axis('off')
    plt.show()

# ============================================================
# 7. QUICK USAGE EXAMPLES (commented—use in exam as needed)
# ============================================================

if __name__=="__main__":
    # --------- 1) Initialize any env ----------
    env=make_env(env_type="frozenlake",size=4,is_slippery=True)
    # env=make_env(env_type="taxi")
    # env=make_env(env_type="cliff")

    # --------- 2) Random policy + iterative policy evaluation ----------
    rand_policy=init_random_policy(env)
    V_rand,_=policy_evaluation_dp(env,rand_policy,discount_factor=0.99,
                                  theta=1e-9,max_iters=50,print_each=True,
                                  return_history=True)
    plot_values(env,V_rand,title="Values under Random Policy")

    # --------- 3) Policy iteration (optimal policy + V) ----------
    opt_policy,opt_V,iters,stable=policy_iteration(env,discount_factor=0.99,
                                                   theta=1e-9,max_iterations=100,
                                                   verbose=True)
    print("Policy iteration stable:",stable,"after",iters,"iterations")
    plot_values(env,opt_V,title="Optimal State Values (Policy Iteration)")
    if env.spec.id.startswith("FrozenLake") or env.spec.id.startswith("CliffWalking"):
        plot_grid_policy(env,opt_V,opt_policy,title="Optimal Policy (Policy Iteration)")

    # --------- 4) Value iteration from scratch ----------
    V_vi,policy_vi=value_iteration(env,gamma=0.99,theta=1e-9,max_iterations=100,
                                   verbose=True)
    plot_values(env,V_vi,title="Optimal State Values (Value Iteration)")
    if env.spec.id.startswith("FrozenLake") or env.spec.id.startswith("CliffWalking"):
        plot_grid_policy(env,V_vi,policy_vi,title="Optimal Policy (Value Iteration)")




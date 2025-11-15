import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP =====================

def make_frozenlake_env(is_slippery=True,render_mode='ansi'):
    """
    Create FrozenLake-v1 environment.
    """
    env=gym.make('FrozenLake-v1',is_slippery=is_slippery,render_mode=render_mode)
    return env

# ===================== 2. Q-FROM-V =====================

def q_from_v(env,V,s,gamma=1.0):
    """
    Compute Q(s,a) for all actions a at state s, given state values V.
    Uses env.unwrapped.P transition model:
      P[s][a] -> list of (prob,next_state,reward,done)
    """
    P=env.unwrapped.P
    nA=env.action_space.n
    q=np.zeros(nA)
    for a in range(nA):
        for prob,next_state,reward,done in P[s][a]:
            q[a]+=prob*(reward+gamma*V[next_state])
    return q

# ===================== 3. POLICY IMPROVEMENT (GREEDY) =====================

def policy_improvement(env,V,discount_factor=1.0):
    """
    Greedy policy improvement:
      π'(s)=argmax_a Q(s,a)
    Returns deterministic policy as one-hot [nS,nA].
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.zeros((nS,nA))
    for s in range(nS):
        Q=q_from_v(env,V,s,discount_factor)
        best_action=int(np.argmax(Q))
        policy[s]=np.eye(nA)[best_action]
    return policy

# ===================== 4. VISUALIZATION (VALUE OR POLICY) =====================

def plot_frozenlake(V,policy,env,draw_vals=True):
    """
    Visualize FrozenLake grid:
      - if draw_vals=True: show V(s) numbers
      - if draw_vals=False: show arrows for best action
    """
    nrow=env.unwrapped.nrow
    ncol=env.unwrapped.ncol
    nA=env.action_space.n
    arrow_symbols={0:'←',1:'↓',2:'→',3:'↑'}

    grid=np.reshape(V,(nrow,ncol))
    plt.figure(figsize=(6,6))
    plt.imshow(grid,cmap='cool',interpolation='none')

    for s in range(nrow*ncol):
        row,col=divmod(s,ncol)
        best_action=int(np.argmax(policy[s]))
        if draw_vals:
            plt.text(col,row,f'{V[s]:.2f}',ha='center',va='center',
                     color='white',fontsize=10)
        else:
            plt.text(col,row,arrow_symbols[best_action],ha='center',va='center',
                     color='white',fontsize=14)

    plt.title("Value Function" if draw_vals else "Policy")
    plt.axis('off')
    plt.show()

# ===================== 5. OPTIONAL: POLICY EVALUATION (FOR POLICY ITERATION) =====================

def policy_evaluation(env,policy,discount_factor=1.0,theta=1e-9):
    """
    Evaluate a given policy π to compute V^π using Bellman expectation.
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

# ===================== 6. OPTIONAL: FULL POLICY ITERATION =====================

def policy_iteration(env,discount_factor=1.0,theta=1e-9,max_iterations=1000,verbose=False):
    """
    Classic Policy Iteration:
      1) Policy Evaluation
      2) Policy Improvement
    until policy is stable.
    Returns (policy,V,num_iterations,stable_flag).
    """
    nS=env.observation_space.n
    nA=env.action_space.n

    # start with random uniform policy
    policy=np.ones((nS,nA))/nA

    for i in range(max_iterations):
        V=policy_evaluation(env,policy,discount_factor,theta)
        new_policy=policy_improvement(env,V,discount_factor)

        stable=np.array_equal(new_policy,policy)
        if verbose:
            print("Iteration",i+1,"stable:",stable)
        policy=new_policy
        if stable:
            return policy,V,i+1,True

    return policy,V,max_iterations,False

# ===================== 7. QUICK DEMO =====================

if __name__=="__main__":
    env=make_frozenlake_env(is_slippery=True,render_mode='ansi')

    # Example: random V, greedy improvement, visualize
    V=np.random.rand(env.observation_space.n)
    policy=policy_improvement(env,V,discount_factor=1.0)
    plot_frozenlake(V,policy,env,draw_vals=True)   # Value Function
    plot_frozenlake(V,policy,env,draw_vals=False)  # Policy arrows

    # Example: full policy iteration (Task 4)
    opt_policy,opt_V,iters,stable=policy_iteration(env,discount_factor=0.99,theta=1e-9,verbose=True)
    print("Policy iteration stable:",stable,"after",iters,"iterations")
    plot_frozenlake(opt_V,opt_policy,env,draw_vals=True)
    plot_frozenlake(opt_V,opt_policy,env,draw_vals=False)

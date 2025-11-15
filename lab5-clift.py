import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP (CLIFFWALKING) =====================

def make_cliff_env(render_mode='ansi'):
    """
    Create CliffWalking-v0 environment (4x12 grid by default).
    """
    env=gym.make('CliffWalking-v0',render_mode=render_mode)
    return env

# ===================== 2. Q-FROM-V FOR CLIFF =====================

def q_from_v_cliff(env,V,s,gamma=1.0):
    """
    Compute Q(s,a) for CliffWalking-v0 given state values V.
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

def policy_improvement_cliff(env,V,discount_factor=1.0):
    """
    Greedy policy improvement for CliffWalking:
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

# ===================== 4. POLICY EVALUATION (FOR CLIFF) =====================

def policy_evaluation_cliff(env,policy,discount_factor=1.0,theta=1e-9):
    """
    Policy evaluation for CliffWalking using Bellman expectation equation.
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

# ===================== 5. FULL POLICY ITERATION (CLIFF) =====================

def policy_iteration_cliff(env,discount_factor=1.0,theta=1e-9,max_iterations=1000,verbose=False):
    """
    Policy Iteration for CliffWalking-v0:
      repeat:
        1) Policy Evaluation
        2) Policy Improvement
      until policy stable.
    Returns (policy,V,num_iterations,stable_flag).
    """
    nS=env.observation_space.n
    nA=env.action_space.n

    policy=np.ones((nS,nA))/nA  # start with uniform random policy

    for i in range(max_iterations):
        V=policy_evaluation_cliff(env,policy,discount_factor,theta)
        new_policy=policy_improvement_cliff(env,V,discount_factor)
        stable=np.array_equal(new_policy,policy)
        if verbose:
            print("Iteration",i+1,"stable:",stable)
        policy=new_policy
        if stable:
            return policy,V,i+1,True

    return policy,V,max_iterations,False

# ===================== 6. PLOT VALUE + POLICY FOR CLIFFWALKING =====================

# Cliff actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
CLIFF_ARROW_DICT={0:'↑',1:'→',2:'↓',3:'←'}

def plot_cliff(V,policy,env,draw_vals=True):
    """
    Visualize CliffWalking grid:
      - S (start), G (goal), C (cliff), . (safe)
      - V(s) values or arrows for greedy actions.
    """
    nrow,ncol=env.unwrapped.shape  # (4,12)
    V_sq=V.reshape((nrow,ncol))

    plt.figure(figsize=(4,2))
    plt.imshow(V_sq,cmap='cool',alpha=0.7)
    ax=plt.gca()

    # grid lines
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

            # tile label
            plt.text(c,r,tile,ha='center',va='center',color=color,fontsize=10,fontweight='bold')

            if draw_vals and tile!='C':
                plt.text(c,r+0.3,f"{val:.1f}",ha='center',va='center',color='black',fontsize=6)
            elif not draw_vals and policy is not None and tile!='C':
                best_action=int(np.argmax(policy[s]))
                arrow=CLIFF_ARROW_DICT.get(best_action,'')
                plt.text(c,r-0.25,arrow,ha='center',va='center',color='purple',fontsize=10)

    plt.title("CliffWalking: "+("Value Function" if draw_vals else "Policy"))
    plt.axis('off')
    plt.show()

# ===================== 7. QUICK DEMO =====================

if __name__=="__main__":
    env=make_cliff_env()

    # Example 1: random V → greedy policy improvement
    V=np.random.rand(env.observation_space.n)
    greedy_policy=policy_improvement_cliff(env,V,discount_factor=0.99)
    print("Greedy policy from random V shape:",greedy_policy.shape)
    plot_cliff(V,greedy_policy,env,draw_vals=True)
    plot_cliff(V,greedy_policy,env,draw_vals=False)

    # Example 2: full policy iteration (optimal policy + V)
    opt_policy,opt_V,iters,stable=policy_iteration_cliff(env,discount_factor=0.99,theta=1e-9,verbose=True)
    print("Cliff policy iteration stable:",stable,"after",iters,"iterations")
    plot_cliff(opt_V,opt_policy,env,draw_vals=True)
    plot_cliff(opt_V,opt_policy,env,draw_vals=False)

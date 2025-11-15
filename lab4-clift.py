import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENV SETUP (CLIFFWALKING) =====================

def make_cliff_env(render_mode='ansi'):
    """
    Create CliffWalking-v0 environment.
    Grid is 4x12 by default.
    """
    env=gym.make('CliffWalking-v0',render_mode=render_mode)
    return env

# ===================== 2. GENERIC VALUE ITERATION (REUSE) =====================

def value_iteration_discrete(env,gamma=0.99,theta=1e-8):
    """
    Value iteration for any discrete env with env.unwrapped.P.
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

def value_iteration_cliff(env,gamma=0.99,theta=1e-8):
    """
    Wrapper for CliffWalking-v0.
    """
    return value_iteration_discrete(env,gamma,theta)

# ===================== 3. DERIVE GREEDY POLICY FROM V =====================

def derive_greedy_policy_discrete(env,V,gamma=0.99):
    """
    Derive greedy deterministic policy for any discrete env.
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

def derive_greedy_policy_cliff(env,V,gamma=0.99):
    return derive_greedy_policy_discrete(env,V,gamma)

# ===================== 4. RANDOM POLICY INIT =====================

def init_random_policy_cliff(env):
    """
    Uniform random policy for CliffWalking-v0.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA
    print("Random Cliff policy shape:",policy.shape)
    return policy

# ===================== 5. POLICY EVALUATION =====================

def policy_evaluation_discrete(env,policy,discount_factor=1.0,theta=1e-9):
    """
    Policy evaluation for any discrete env with P.
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

def policy_evaluation_cliff(env,policy,discount_factor=0.99,theta=1e-9):
    return policy_evaluation_discrete(env,policy,discount_factor,theta)

# ===================== 6. PLOT VALUE + POLICY FOR CLIFFWALKING =====================

# Cliff actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
CLIFF_ARROW_DICT={0:'↑',1:'→',2:'↓',3:'←'}

def plot_cliff(env,V,policy=None,col_ramp=1,dpi=175,draw_vals=False):
    """
    Visualize CliffWalking grid with:
      - start (S), goal (G), cliff (C), safe cells (.)
      - state values
      - policy arrows
    """
    plt.rcParams['figure.dpi']=dpi
    plt.rcParams.update({'axes.edgecolor':(0.32,0.36,0.38)})

    nrow,ncol=env.unwrapped.shape  # (4,12)
    V_sq=V.reshape((nrow,ncol))

    plt.figure(figsize=(4,2))
    plt.imshow(V_sq,cmap='cool' if col_ramp else 'gray',alpha=0.7)
    ax=plt.gca()

    # grid lines
    for x in range(ncol+1):
        ax.axvline(x-0.5,lw=0.5,color='black')
    for y in range(nrow+1):
        ax.axhline(y-0.5,lw=0.5,color='black')

    # define special cells
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

            # value text
            if draw_vals and tile!='C':
                plt.text(c,r+0.3,f"{val:.1f}",ha='center',va='center',color='black',fontsize=6)

            # arrow from policy
            if policy is not None and tile!='C':
                best_action=int(np.argmax(policy[s]))
                arrow=CLIFF_ARROW_DICT.get(best_action,'')
                plt.text(c,r-0.25,arrow,ha='center',va='center',color='purple',fontsize=10)

    plt.title("CliffWalking: Policy and State Values")
    plt.axis('off')
    plt.show()

# ===================== 7. QUICK DEMO WORKFLOW =====================

if __name__=="__main__":
    env=make_cliff_env()

    # --- Random policy + evaluation ---
    rand_policy=init_random_policy_cliff(env)
    V_rand=policy_evaluation_cliff(env,rand_policy,discount_factor=0.99,theta=1e-9)
    print("Random policy V (summary): min={:.2f}, max={:.2f}, mean={:.2f}"
          .format(np.min(V_rand),np.max(V_rand),np.mean(V_rand)))
    plot_cliff(env,V_rand,rand_policy,draw_vals=False)

    # --- Optimal policy via value iteration ---
    V_opt=value_iteration_cliff(env,gamma=0.99,theta=1e-8)
    opt_policy=derive_greedy_policy_cliff(env,V_opt,gamma=0.99)
    print("Optimal V (summary): min={:.2f}, max={:.2f}, mean={:.2f}"
          .format(np.min(V_opt),np.max(V_opt),np.mean(V_opt)))
    plot_cliff(env,V_opt,opt_policy,draw_vals=True)

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

# ===================== 2. VALUE ITERATION (OPTIMAL V) =====================

def value_iteration_frozenlake(env,gamma=0.99,theta=1e-8):
    """
    Compute optimal state-value function V using value iteration.
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

# ===================== 3. DERIVE OPTIMAL POLICY FROM V =====================

def derive_greedy_policy(env,V,gamma=0.99):
    """
    Derive greedy (deterministic) policy π*(s) from value function V.
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

# ===================== 4. RANDOM POLICY INITIALISATION =====================

def init_random_policy(env):
    """
    Create a uniform random policy over actions for all states.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    policy=np.ones((nS,nA))/nA
    print("Random policy shape:",policy.shape)
    return policy

# ===================== 5. POLICY EVALUATION =====================

def policy_evaluation(env,policy,discount_factor=1.0,theta=1e-9,draw=False):
    """
    Evaluate a given policy π(s,a) to compute V^π using Bellman expectation.
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

    if draw:
        side=int(np.sqrt(nS))
        print("Value function grid:")
        print(V.reshape(side,side))
    return V

# ===================== 6. PLOTTING VALUE FUNCTION & POLICY =====================

def plot_frozenlake(env,V,policy,col_ramp=1,dpi=175,draw_vals=False):
    """
    Visualize FrozenLake grid with:
      - tiles (S, F, H, G)
      - state values V
      - policy arrows (if policy not None)
    """
    plt.rcParams['figure.dpi']=dpi
    plt.rcParams.update({'axes.edgecolor':(0.32,0.36,0.38)})
    plt.rcParams.update({'font.size':6 if env.unwrapped.nrow==8 else 8})

    desc=env.unwrapped.desc
    nrow,ncol=desc.shape
    V_sq=V.reshape((nrow,ncol))

    plt.figure(figsize=(3,3))
    plt.imshow(V_sq,cmap='cool' if col_ramp else 'gray',alpha=0.7)
    ax=plt.gca()

    arrow_dict={
        0:'←', # LEFT
        1:'↓', # DOWN
        2:'→', # RIGHT
        3:'↑'  # UP
    }

    # grid lines
    for x in range(ncol+1):
        ax.axvline(x-0.5,lw=0.5,color='black')
    for y in range(nrow+1):
        ax.axhline(y-0.5,lw=0.5,color='black')

    # fill cells
    for r in range(nrow):
        for c in range(ncol):
            s=r*ncol+c
            val=V[s]
            tile=desc[r,c].decode('utf-8')

            if tile=='H':
                color='red'
            elif tile=='G':
                color='green'
            elif tile=='S':
                color='blue'
            else:
                color='black'

            # tile letter
            plt.text(c,r,tile,ha='center',va='center',color=color,fontsize=10,fontweight='bold')

            # value text
            if draw_vals and tile not in ['H']:
                plt.text(c,r+0.3,f"{val:.2f}",ha='center',va='center',color='black',fontsize=6)

            # arrow for policy
            if policy is not None:
                best_action=int(np.argmax(policy[s]))
                plt.text(c,r-0.25,arrow_dict[best_action],ha='center',va='center',
                         color='purple',fontsize=12)

    plt.title("FrozenLake: Policy and State Values")
    plt.axis('off')
    plt.show()

# ===================== 7. QUICK DEMO / WORKFLOW =====================

if __name__=="__main__":
    # 1) make env
    env=make_frozenlake_env(is_slippery=True,render_mode='ansi')

    # 2) random policy + evaluation
    rand_policy=init_random_policy(env)
    V_rand=policy_evaluation(env,rand_policy,discount_factor=0.99,theta=1e-9,draw=True)
    plot_frozenlake(env,V_rand,rand_policy,draw_vals=True)

    # 3) optimal V via value iteration + greedy policy
    V_opt=value_iteration_frozenlake(env,gamma=0.99,theta=1e-8)
    opt_policy=derive_greedy_policy(env,V_opt,gamma=0.99)
    plot_frozenlake(env,V_opt,opt_policy,draw_vals=True)


# With this, for Lab 4 you can:

# Show policy evaluation:

env=make_frozenlake_env()
policy=init_random_policy(env)
V=policy_evaluation(env,policy,discount_factor=0.99,draw=True)
plot_frozenlake(env,V,policy,draw_vals=True)


# Show optimal value iteration result (foundation for later labs on policy iteration/value iteration):

V_opt=value_iteration_frozenlake(env,gamma=0.99)
opt_policy=derive_greedy_policy(env,V_opt,gamma=0.99)
plot_frozenlake(env,V_opt,opt_policy,draw_vals=True)
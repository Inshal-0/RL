import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP =====================

def make_frozenlake_env(is_slippery=False,render_mode=None):
    """
    Create FrozenLake-v1 environment (default 4x4) for MC/TD lab.
    is_slippery=False -> deterministic dynamics (easier to analyze).
    """
    env=gym.make("FrozenLake-v1",is_slippery=is_slippery,render_mode=render_mode)
    return env

# ===================== 2. POLICY HELPERS (RANDOM & SIMPLE DETERMINISTIC) =====================

def make_random_policy(env):
    """
    Returns a dict policy[state]=action with random fixed action per state.
    """
    nS=env.observation_space.n
    policy={s:np.random.choice([0,1,2,3]) for s in range(nS)}
    return policy

def make_deterministic_policy(env,mode="right"):
    """
    Simple deterministic policy:
      mode="right" -> always action 2
      mode="down"  -> always action 1
    """
    nS=env.observation_space.n
    if mode=="right":
        a=2
    elif mode=="down":
        a=1
    else:
        a=0
    policy={s:a for s in range(nS)}
    return policy

# ===================== 3. MONTE CARLO PREDICTION (FIRST-VISIT) =====================

def mc_prediction(env,policy,episodes=10000,gamma=0.99):
    """
    First-visit Monte Carlo prediction:
    Estimates V(s) for a given deterministic policy.
    Returns:
      V: final value function
      V_track: list of V snapshots after each episode
      rewards_per_episode: list of total (undiscounted) rewards per episode
    """
    nS=env.observation_space.n
    V=np.zeros(nS)
    returns={s:[] for s in range(nS)}
    V_track=[]
    rewards_per_episode=[]

    for ep in range(episodes):
        episode=[]
        state,_=env.reset()
        done=False
        total_reward=0

        # generate full episode
        while not done:
            action=policy[state]
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            episode.append((state,reward))
            total_reward+=reward
            state=next_state

        # compute returns backwards
        G=0
        visited_states=set()
        for s,r in reversed(episode):
            G=gamma*G+r
            if s not in visited_states:
                returns[s].append(G)
                V[s]=np.mean(returns[s])
                visited_states.add(s)

        V_track.append(V.copy())
        rewards_per_episode.append(total_reward)

    return V,V_track,rewards_per_episode

# ===================== 4. TEMPORAL DIFFERENCE PREDICTION (TD(0)) =====================

def td_prediction(env,policy,episodes=10000,alpha=0.1,gamma=0.99):
    """
    TD(0) prediction:
    Online, bootstrapping update after each step.
    Returns:
      V: final value function
      V_track: list of V snapshots after each episode
      rewards_per_episode: list of total (undiscounted) rewards per episode
    """
    nS=env.observation_space.n
    V=np.zeros(nS)
    V_track=[]
    rewards_per_episode=[]

    for ep in range(episodes):
        state,_=env.reset()
        done=False
        total_reward=0

        while not done:
            action=policy[state]
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated

            # TD(0) update
            V[state]=V[state]+alpha*(reward+gamma*V[next_state]-V[state])

            total_reward+=reward
            state=next_state

        V_track.append(V.copy())
        rewards_per_episode.append(total_reward)

    return V,V_track,rewards_per_episode

# ===================== 5. PLOTTING HELPERS =====================

def plot_convergence(V_track,title="Value Estimates Over Episodes"):
    """
    Plot V(s) over episodes for each state.
    """
    nS=len(V_track[0])
    plt.figure(figsize=(8,5))
    for s in range(nS):
        values=[v[s] for v in V_track]
        plt.plot(values,label=f"State {s}")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Value Estimate V(s)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_avg_reward(rewards_mc,rewards_td,window=100):
    """
    Plot moving average of episode rewards for MC and TD.
    """
    def moving_avg(x,w):
        if len(x)<w:
            return x
        return np.convolve(x,np.ones(w)/w,mode='valid')

    avg_mc=moving_avg(rewards_mc,window)
    avg_td=moving_avg(rewards_td,window)

    plt.figure(figsize=(8,5))
    plt.plot(avg_mc,label=f"MC (window={window})")
    plt.plot(avg_td,label=f"TD(0) (window={window})")
    plt.title("Average Episode Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 6. EXPERIMENT RUNNER (MC vs TD) =====================

def run_mc_td_experiment(episodes=5000,gamma=0.99,alpha=0.1,
                         policy_mode="random",is_slippery=False):
    """
    Full lab-style experiment:
      - make env
      - choose policy (random or deterministic)
      - run MC and TD(0)
      - print final V
      - plot convergence and avg reward
    """
    env=make_frozenlake_env(is_slippery=is_slippery,render_mode=None)

    if policy_mode=="random":
        policy=make_random_policy(env)
    else:
        policy=make_deterministic_policy(env,mode=policy_mode)

    print("Using policy_mode:",policy_mode,"is_slippery:",is_slippery)

    # MC
    V_mc,V_mc_track,rewards_mc=mc_prediction(env,policy,episodes=episodes,gamma=gamma)

    # TD(0)
    V_td,V_td_track,rewards_td=td_prediction(env,policy,episodes=episodes,
                                             alpha=alpha,gamma=gamma)

    print("\nMonte Carlo Value Function (rounded):")
    print(np.round(V_mc,3))
    print("\nTD(0) Value Function (rounded):")
    print(np.round(V_td,3))

    # Convergence plots
    plot_convergence(V_mc_track,"Monte Carlo Value Estimates Over Episodes")
    plot_convergence(V_td_track,"TD(0) Value Estimates Over Episodes")

    # Average reward plots
    plot_avg_reward(rewards_mc,rewards_td,window=100)

    return {
        "V_mc":V_mc,
        "V_td":V_td,
        "V_mc_track":V_mc_track,
        "V_td_track":V_td_track,
        "rewards_mc":rewards_mc,
        "rewards_td":rewards_td,
        "policy":policy
    }

# ===================== 7. MAIN (OPTIONAL DEMO) =====================

if __name__=="__main__":
    # Example: random policy, deterministic env, 5000 episodes
    results=run_mc_td_experiment(episodes=5000,gamma=0.99,alpha=0.1,
                                 policy_mode="random",is_slippery=False)

    # Example: deterministic "right" policy
    # results=run_mc_td_experiment(episodes=5000,gamma=0.99,alpha=0.1,
    #                              policy_mode="right",is_slippery=False)

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP =====================

def make_cliff_env(render_mode=None):
    env=gym.make("CliffWalking-v1",render_mode=render_mode)
    return env

def describe_cliff_env(env):
    print("CliffWalking-v1:")
    print("  States:",env.observation_space.n,"(4x12 grid)")
    print("  Actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT")
    print("  Rewards: -1 per step, -100 on cliff, 0 at goal")

# ===================== 2. POLICY HELPERS =====================

def make_random_policy_cliff(env):
    nS=env.observation_space.n
    nA=env.action_space.n
    policy={s:np.random.choice(np.arange(nA)) for s in range(nS)}
    return policy

def make_simple_policy_cliff(env,mode="right"):
    """
    mode options:
      'right' -> always 1
      'down'  -> always 2
      'up'    -> always 0
      'left'  -> always 3
    """
    nS=env.observation_space.n
    a_map={"up":0,"right":1,"down":2,"left":3}
    a=a_map.get(mode,1)
    policy={s:a for s in range(nS)}
    return policy

# ===================== 3. MONTE CARLO PREDICTION =====================

def mc_prediction_cliff(env,policy,episodes=5000,gamma=0.99,max_steps=500):
    """
    First-visit MC prediction for CliffWalking-v1.
    Returns V,V_track,rewards_per_episode.
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
        steps=0

        while not done and steps<max_steps:
            action=policy[state]
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            episode.append((state,reward))
            total_reward+=reward
            state=next_state
            steps+=1

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

# ===================== 4. TD(0) PREDICTION =====================

def td_prediction_cliff(env,policy,episodes=5000,alpha=0.1,gamma=0.99,max_steps=500):
    """
    TD(0) evaluation for CliffWalking-v1.
    Returns V,V_track,rewards_per_episode.
    """
    nS=env.observation_space.n
    V=np.zeros(nS)
    V_track=[]
    rewards_per_episode=[]

    for ep in range(episodes):
        state,_=env.reset()
        done=False
        total_reward=0
        steps=0

        while not done and steps<max_steps:
            action=policy[state]
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated

            V[state]=V[state]+alpha*(reward+gamma*V[next_state]-V[state])

            total_reward+=reward
            state=next_state
            steps+=1

        V_track.append(V.copy())
        rewards_per_episode.append(total_reward)

    return V,V_track,rewards_per_episode

# ===================== 5. PLOTTING HELPERS =====================

def plot_convergence_cliff(V_track,title="Cliff: Value Estimates Over Episodes",max_states=8):
    """
    Plot V(s) over episodes for a subset of states.
    """
    nS=len(V_track[0])
    states_to_plot=list(range(min(max_states,nS)))
    plt.figure(figsize=(8,5))
    for s in states_to_plot:
        values=[v[s] for v in V_track]
        plt.plot(values,label=f"State {s}")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("V(s)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_avg_reward_cliff(rewards_mc,rewards_td,window=100):
    def moving_avg(x,w):
        if len(x)<w:
            return x
        return np.convolve(x,np.ones(w)/w,mode='valid')

    avg_mc=moving_avg(rewards_mc,window)
    avg_td=moving_avg(rewards_td,window)

    plt.figure(figsize=(8,5))
    plt.plot(avg_mc,label=f"MC (window={window})")
    plt.plot(avg_td,label=f"TD(0) (window={window})")
    plt.title("Cliff: Average Episode Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 6. FULL EXPERIMENT RUNNER =====================

def run_mc_td_experiment_cliff(episodes=5000,gamma=0.99,alpha=0.1,
                               policy_mode="random"):
    """
    Lab 7 experiment on CliffWalking-v1:
      - choose policy (random or simple deterministic)
      - run MC + TD(0)
      - print final V
      - plot convergence + avg rewards
    """
    env=make_cliff_env()
    describe_cliff_env(env)

    if policy_mode=="random":
        policy=make_random_policy_cliff(env)
    else:
        policy=make_simple_policy_cliff(env,mode=policy_mode)

    print("\nUsing policy_mode:",policy_mode)

    V_mc,V_mc_track,rewards_mc=mc_prediction_cliff(env,policy,episodes=episodes,gamma=gamma)
    V_td,V_td_track,rewards_td=td_prediction_cliff(env,policy,episodes=episodes,alpha=alpha,gamma=gamma)

    print("\nMonte Carlo Value Function (rounded):")
    print(np.round(V_mc,1))
    print("\nTD(0) Value Function (rounded):")
    print(np.round(V_td,1))

    plot_convergence_cliff(V_mc_track,"Cliff: MC Value Estimates Over Episodes")
    plot_convergence_cliff(V_td_track,"Cliff: TD(0) Value Estimates Over Episodes")
    plot_avg_reward_cliff(rewards_mc,rewards_td,window=100)

    return {
        "V_mc":V_mc,
        "V_td":V_td,
        "V_mc_track":V_mc_track,
        "V_td_track":V_td_track,
        "rewards_mc":rewards_mc,
        "rewards_td":rewards_td,
        "policy":policy
    }

if __name__=="__main__":
    # Example usage:
    # results_cliff=run_mc_td_experiment_cliff(episodes=5000,gamma=0.99,alpha=0.1,policy_mode="random")
    pass

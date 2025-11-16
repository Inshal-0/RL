import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP =====================

def make_taxi_env(render_mode=None):
    env=gym.make("Taxi-v3",render_mode=render_mode)
    return env

def describe_taxi_env(env):
    print("Taxi-v3:")
    print("  States:",env.observation_space.n)
    print("  Actions: 0=South,1=North,2=East,3=West,4=Pickup,5=Dropoff")
    print("  Rewards: +20 success, -1 step, -10 illegal pickup/dropoff")

# ===================== 2. POLICY HELPERS =====================

def make_random_policy_taxi(env):
    nS=env.observation_space.n
    nA=env.action_space.n
    policy={s:np.random.choice(np.arange(nA)) for s in range(nS)}
    return policy

def make_simple_policy_taxi(env,mode="south"):
    """
    mode options:
      'south' -> always 0
      'north' -> always 1
      'east'  -> always 2
      'west'  -> always 3
    """
    nS=env.observation_space.n
    a_map={"south":0,"north":1,"east":2,"west":3}
    a=a_map.get(mode,0)
    policy={s:a for s in range(nS)}
    return policy

# ===================== 3. MONTE CARLO PREDICTION (FIRST-VISIT) =====================

def mc_prediction_taxi(env,policy,episodes=5000,gamma=0.99,max_steps=200):
    """
    First-visit Monte Carlo for Taxi-v3 under deterministic policy.
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

def td_prediction_taxi(env,policy,episodes=5000,alpha=0.1,gamma=0.99,max_steps=200):
    """
    TD(0) policy evaluation for Taxi-v3.
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

def plot_convergence_taxi(V_track,title="Taxi: Value Estimates Over Episodes",max_states=5):
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

def plot_avg_reward_taxi(rewards_mc,rewards_td,window=100):
    def moving_avg(x,w):
        if len(x)<w:
            return x
        return np.convolve(x,np.ones(w)/w,mode='valid')

    avg_mc=moving_avg(rewards_mc,window)
    avg_td=moving_avg(rewards_td,window)

    plt.figure(figsize=(8,5))
    plt.plot(avg_mc,label=f"MC (window={window})")
    plt.plot(avg_td,label=f"TD(0) (window={window})")
    plt.title("Taxi: Average Episode Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 6. FULL EXPERIMENT RUNNER (LAB STYLE) =====================

def run_mc_td_experiment_taxi(episodes=5000,gamma=0.99,alpha=0.1,
                              policy_mode="random"):
    """
    Lab 7 experiment on Taxi-v3:
      - choose policy (random or simple deterministic)
      - run MC + TD(0)
      - print final V
      - plot convergence + avg reward
    """
    env=make_taxi_env()
    describe_taxi_env(env)

    if policy_mode=="random":
        policy=make_random_policy_taxi(env)
    else:
        policy=make_simple_policy_taxi(env,mode=policy_mode)

    print("\nUsing policy_mode:",policy_mode)

    V_mc,V_mc_track,rewards_mc=mc_prediction_taxi(env,policy,episodes=episodes,gamma=gamma)
    V_td,V_td_track,rewards_td=td_prediction_taxi(env,policy,episodes=episodes,alpha=alpha,gamma=gamma)

    print("\nMonte Carlo Value Function (rounded):")
    print(np.round(V_mc,3))
    print("\nTD(0) Value Function (rounded):")
    print(np.round(V_td,3))

    plot_convergence_taxi(V_mc_track,"Taxi: MC Value Estimates Over Episodes")
    plot_convergence_taxi(V_td_track,"Taxi: TD(0) Value Estimates Over Episodes")
    plot_avg_reward_taxi(rewards_mc,rewards_td,window=100)

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
    # results_taxi=run_mc_td_experiment_taxi(episodes=5000,gamma=0.99,alpha=0.1,policy_mode="random")
    pass

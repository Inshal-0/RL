import gym
import numpy as np
import matplotlib.pyplot as plt

# --------- Helper: Step that works for both 4-return and 5-return envs ---------
def step_env(env,action):
    """Wrapper to handle both Gym (4 values) and Gymnasium (5 values)."""
    result=env.step(action)
    if len(result)==4:
        next_state,reward,done,info=result
        terminated,truncated=done,False
    else:
        next_state,reward,terminated,truncated,info=result
    done=terminated or truncated
    return next_state,reward,terminated,truncated,done,info

# --------- 1. Make deterministic / stochastic FrozenLake environments ---------
def make_frozenlake(det=True,size=4):
    """det=True -> is_slippery=False (deterministic), det=False -> stochastic."""
    env=gym.make('FrozenLake-v1',is_slippery=not det)
    return env

# --------- 2. Explore environment: state, action space, observation space ---------
def explore_env(env):
    state=env.reset()
    print("Initial State:",state)
    print("Action Space:",env.action_space)
    print("Observation Space:",env.observation_space)

# --------- 3. Basic interaction loop with random policy (single episode) ---------
def run_random_episode(env,render=False,max_steps=100):
    state=env.reset()
    terminated=False
    truncated=False
    total_reward=0
    step_count=0

    while not (terminated or truncated) and step_count<max_steps:
        if render:
            env.render()
        action=env.action_space.sample()
        next_state,reward,terminated,truncated,done,info=step_env(env,action)
        print(f"Step {step_count}: State={state}, Action={action}, Reward={reward}, Next State={next_state}, Terminated={terminated}")
        total_reward+=reward
        state=next_state
        step_count+=1

    print(f"Episode ended with total reward: {total_reward}\n")
    return total_reward

# --------- 4. Run multiple random episodes and return rewards ---------
def run_random_episodes(env,num_episodes=100,max_steps=100):
    rewards=[]
    for ep in range(num_episodes):
        state=env.reset()
        terminated=False
        truncated=False
        total_reward=0
        step_count=0

        while not (terminated or truncated) and step_count<max_steps:
            action=env.action_space.sample()
            next_state,reward,terminated,truncated,done,info=step_env(env,action)
            total_reward+=reward
            state=next_state
            step_count+=1

        print(f"Episode {ep+1} ended with total reward: {total_reward}")
        rewards.append(total_reward)
    return rewards

# --------- 5. Visualize path on grid ---------
def visualize_path(path,size=4):
    grid=np.full((size,size),'-',dtype=object)
    for step,state in enumerate(path):
        row,col=divmod(state,size)
        grid[row,col]=str(step)
    print(grid)

def random_path_episode(env,size=4,max_steps=100):
    state=env.reset()
    terminated=False
    truncated=False
    path=[state]
    step_count=0

    while not (terminated or truncated) and step_count<max_steps:
        action=env.action_space.sample()
        state,reward,terminated,truncated,done,info=step_env(env,action)
        path.append(state)
        step_count+=1

    visualize_path(path,size=size)
    return path

# --------- 6. Compare deterministic vs stochastic environments ---------
def compare_det_vs_stoch(num_episodes=20,max_steps=100):
    env_det=make_frozenlake(det=True)
    env_stoch=make_frozenlake(det=False)

    print("Running deterministic environment...")
    rewards_det=run_random_episodes(env_det,num_episodes,max_steps)
    print("\nRunning stochastic environment...")
    rewards_stoch=run_random_episodes(env_stoch,num_episodes,max_steps)

    print("\nDeterministic total rewards:",rewards_det)
    print("Stochastic total rewards:",rewards_stoch)

    return rewards_det,rewards_stoch

# --------- 7. Track cumulative rewards and plot ---------
def plot_random_policy_rewards(env,num_episodes=10,max_steps=100):
    rewards=[]
    for ep in range(num_episodes):
        state=env.reset()
        terminated=False
        truncated=False
        total_reward=0
        step_count=0

        while not (terminated or truncated) and step_count<max_steps:
            action=env.action_space.sample()
            state,reward,terminated,truncated,done,info=step_env(env,action)
            total_reward+=reward
            step_count+=1

        rewards.append(total_reward)

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Random Policy Reward per Episode")
    plt.show()
    return rewards

# --------- 8. Manual (fixed) action policy ---------
def run_manual_policy(env,actions,num_episodes=5,max_steps=None):
    if max_steps is None:
        max_steps=len(actions)
    for ep in range(num_episodes):
        state=env.reset()
        total_reward=0
        terminated=False
        truncated=False
        print(f"\nEpisode {ep+1}")
        step_idx=0

        while not (terminated or truncated) and step_idx<max_steps and step_idx<len(actions):
            action=actions[step_idx]
            next_state,reward,terminated,truncated,done,info=step_env(env,action)
            print(f"  Step {step_idx}: State={state}, Action={action}, Next State={next_state}, Reward={reward}")
            total_reward+=reward
            state=next_state
            step_idx+=1

        print(f"Total reward with manual policy: {total_reward}")

# --------- 9. Try another environment (Taxi-v3 / CliffWalking-v0) ---------
def explore_other_env(env_name='Taxi-v3',num_episodes=5,max_steps=200):
    env=gym.make(env_name)
    print("Env name:",env_name)
    print("Action Space:",env.action_space)
    print("Observation Space:",env.observation_space)

    for ep in range(num_episodes):
        state=env.reset()
        if isinstance(state,tuple):  # gymnasium sometimes returns (obs,info)
            state,_=state
        terminated=False
        truncated=False
        total_reward=0
        step_count=0

        while not (terminated or truncated) and step_count<max_steps:
            action=env.action_space.sample()
            next_state,reward,terminated,truncated,done,info=step_env(env,action)
            total_reward+=reward
            state=next_state
            step_count+=1

        print(f"Episode {ep+1} in {env_name}: total_reward={total_reward}")
    return env

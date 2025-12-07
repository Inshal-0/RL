import numpy as np
import gym
import matplotlib.pyplot as plt

def create_taxi_env():
    return gym.make("Taxi-v3")

def target_policy_greedy_row(q_row):
    return np.argmax(q_row)

def init_state_weighted_taxi(Q,C,state,n_actions):
    if state not in Q:
        Q[state]=np.zeros(n_actions)
        C[state]=np.zeros(n_actions)

def init_state_ordinary_taxi(Q,N,state,n_actions):
    if state not in Q:
        Q[state]=np.zeros(n_actions)
        N[state]=np.zeros(n_actions)

def behavior_policy_taxi_uniform(env):
    return env.action_space.sample()

def generate_episode_taxi(env):
    episode=[]
    state=env.reset()[0]
    done=False
    while not done:
        action=behavior_policy_taxi_uniform(env)
        next_state,reward,terminated,truncated,_=env.step(action)
        done=terminated or truncated
        episode.append((state,action,reward))
        state=next_state
    return episode

def off_policy_mc_weighted_taxi(env,episodes,gamma=0.99):
    n_actions=env.action_space.n
    Q={}
    C={}
    for _ in range(episodes):
        episode=generate_episode_taxi(env)
        G=0.0
        W=1.0
        for (s,a,r) in reversed(episode):
            G=gamma*G+r
            init_state_weighted_taxi(Q,C,s,n_actions)
            q=Q[s]
            C[s][a]+=W
            q[a]+= (W/C[s][a])*(G-q[a])
            if target_policy_greedy_row(q)!=a:
                break
            W*=n_actions
    return Q

def off_policy_mc_ordinary_taxi(env,episodes,gamma=0.99):
    n_actions=env.action_space.n
    Q={}
    N={}
    for _ in range(episodes):
        episode=generate_episode_taxi(env)
        G=0.0
        W=1.0
        for (s,a,r) in reversed(episode):
            G=gamma*G+r
            init_state_ordinary_taxi(Q,N,s,n_actions)
            N[s][a]+=1.0
            Q[s][a]+= (W*G-Q[s][a])/N[s][a]
            if target_policy_greedy_row(Q[s])!=a:
                break
            W*=n_actions
    return Q

def compare_mc_variance_taxi(env,episodes,state_action,runs=10,gamma=0.99):
    s_target,a_target=state_action
    ordinary_values=[]
    weighted_values=[]
    for _ in range(runs):
        Q_ord=off_policy_mc_ordinary_taxi(env,episodes,gamma)
        Q_w=off_policy_mc_weighted_taxi(env,episodes,gamma)
        if s_target in Q_ord:
            ordinary_values.append(Q_ord[s_target][a_target])
        if s_target in Q_w:
            weighted_values.append(Q_w[s_target][a_target])
    ordinary_values=np.array(ordinary_values)
    weighted_values=np.array(weighted_values)
    return ordinary_values,weighted_values

def behavior_policy_epsilon_greedy_taxi(Q,state,epsilon,env):
    if np.random.rand()<epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

def q_learning_taxi(env,alpha,gamma,epsilon,episodes,max_steps_per_episode=500):
    n_states=env.observation_space.n
    n_actions=env.action_space.n
    Q=np.zeros((n_states,n_actions))
    rewards=[]
    for _ in range(episodes):
        s=env.reset()[0]
        terminated=False
        truncated=False
        total_reward=0.0
        steps=0
        while not(terminated or truncated):
            a=behavior_policy_epsilon_greedy_taxi(Q,s,epsilon,env)
            s2,r,terminated,truncated,_=env.step(a)
            best_next=np.max(Q[s2])
            Q[s,a]+=alpha*(r+gamma*best_next-Q[s,a])
            s=s2
            total_reward+=r
            steps+=1
            if steps>=max_steps_per_episode:
                break
        rewards.append(total_reward)
    return Q,rewards

def q_learning_taxi_epsilon_sweep(env,epsilons,alpha,gamma,episodes,max_steps_per_episode=500):
    all_rewards=[]
    labels=[]
    for eps in epsilons:
        Q,rewards=q_learning_taxi(env,alpha,gamma,eps,episodes,max_steps_per_episode)
        all_rewards.append(rewards)
        labels.append(f"epsilon={eps}")
    for rewards,label in zip(all_rewards,labels):
        plt.plot(rewards,label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning on Taxi-v3 for different epsilons")
    plt.legend()
    plt.show()
    return all_rewards,labels

def q_learning_taxi_with_policy_logging(env,alpha,gamma,epsilon,episodes,max_steps_per_episode=500):
    n_states=env.observation_space.n
    n_actions=env.action_space.n
    Q=np.zeros((n_states,n_actions))
    rewards=[]
    behavior_actions=[]
    target_actions=[]
    for _ in range(episodes):
        s=env.reset()[0]
        terminated=False
        truncated=False
        total_reward=0.0
        steps=0
        while not(terminated or truncated):
            a=behavior_policy_epsilon_greedy_taxi(Q,s,epsilon,env)
            greedy_a=np.argmax(Q[s])
            behavior_actions.append(a)
            target_actions.append(greedy_a)
            s2,r,terminated,truncated,_=env.step(a)
            best_next=np.max(Q[s2])
            Q[s,a]+=alpha*(r+gamma*best_next-Q[s,a])
            s=s2
            total_reward+=r
            steps+=1
            if steps>=max_steps_per_episode:
                break
        rewards.append(total_reward)
    behavior_actions=np.array(behavior_actions)
    target_actions=np.array(target_actions)
    mismatches=(behavior_actions!=target_actions).sum()
    total=len(behavior_actions)
    mismatch_ratio=mismatches/total if total>0 else 0.0
    return Q,rewards,behavior_actions,target_actions,mismatch_ratio

def extract_greedy_policy_from_Q_taxi(Q):
    return np.argmax(Q,axis=1)

def run_full_taxi_experiment(mc_episodes,q_episodes,alpha,gamma,epsilon):
    env_taxi=create_taxi_env()
    Q_mc_weighted=off_policy_mc_weighted_taxi(env_taxi,mc_episodes,gamma)
    Q_mc_ordinary=off_policy_mc_ordinary_taxi(env_taxi,mc_episodes,gamma)
    Q_q,rew_q=q_learning_taxi(env_taxi,alpha,gamma,epsilon,q_episodes)
    return Q_mc_weighted,Q_mc_ordinary,Q_q,rew_q

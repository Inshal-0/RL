import numpy as np
import gym
import matplotlib.pyplot as plt

def create_blackjack_env():
    return gym.make("Blackjack-v1")

def create_frozenlake_env():
    return gym.make("FrozenLake-v1",is_slippery=False)

def init_state_weighted(Q,C,state,n_actions=2):
    if state not in Q:
        Q[state]=np.zeros(n_actions)
        C[state]=np.zeros(n_actions)

def init_state_ordinary(Q,N,state,n_actions=2):
    if state not in Q:
        Q[state]=np.zeros(n_actions)
        N[state]=np.zeros(n_actions)

def behavior_policy_blackjack():
    return np.random.choice([0,1])

def target_policy_greedy(q_row):
    return np.argmax(q_row)

def generate_episode_blackjack(env):
    episode=[]
    state=env.reset()[0]
    done=False
    while not done:
        action=behavior_policy_blackjack()
        next_state,reward,terminated,truncated,_=env.step(action)
        done=terminated or truncated
        episode.append((state,action,reward))
        state=next_state
    return episode

def off_policy_mc_weighted_blackjack(env,episodes,gamma=1.0):
    Q={}
    C={}
    for _ in range(episodes):
        episode=generate_episode_blackjack(env)
        G=0.0
        W=1.0
        for (s,a,r) in reversed(episode):
            G=gamma*G+r
            init_state_weighted(Q,C,s)
            q=Q[s]
            C[s][a]+=W
            q[a]+= (W/C[s][a])*(G-q[a])
            if target_policy_greedy(q)!=a:
                break
            W*=2.0
    return Q

def off_policy_mc_ordinary_blackjack(env,episodes,gamma=1.0):
    Q={}
    N={}
    for _ in range(episodes):
        episode=generate_episode_blackjack(env)
        G=0.0
        W=1.0
        for (s,a,r) in reversed(episode):
            G=gamma*G+r
            init_state_ordinary(Q,N,s)
            N[s][a]+=1.0
            Q[s][a]+= (W*G-Q[s][a])/N[s][a]
            if target_policy_greedy(Q[s])!=a:
                break
            W*=2.0
    return Q

def compare_mc_variance_blackjack(env,episodes,state_action,runs=10,gamma=1.0):
    s_target,a_target=state_action
    ordinary_values=[]
    weighted_values=[]
    for _ in range(runs):
        Q_ord=off_policy_mc_ordinary_blackjack(env,episodes,gamma)
        Q_w=off_policy_mc_weighted_blackjack(env,episodes,gamma)
        if s_target in Q_ord:
            ordinary_values.append(Q_ord[s_target][a_target])
        if s_target in Q_w:
            weighted_values.append(Q_w[s_target][a_target])
    ordinary_values=np.array(ordinary_values)
    weighted_values=np.array(weighted_values)
    return ordinary_values,weighted_values

def behavior_policy_epsilon_greedy(Q,state,epsilon,env):
    if np.random.rand()<epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

def q_learning_frozenlake(env,alpha,gamma,epsilon,episodes):
    n_states=env.observation_space.n
    n_actions=env.action_space.n
    Q=np.zeros((n_states,n_actions))
    rewards=[]
    for _ in range(episodes):
        s=env.reset()[0]
        terminated=False
        truncated=False
        total_reward=0.0
        while not(terminated or truncated):
            a=behavior_policy_epsilon_greedy(Q,s,epsilon,env)
            s2,r,terminated,truncated,_=env.step(a)
            best_next=np.max(Q[s2])
            Q[s,a]+=alpha*(r+gamma*best_next-Q[s,a])
            s=s2
            total_reward+=r
        rewards.append(total_reward)
    return Q,rewards

def q_learning_epsilon_sweep(env,epsilons,alpha,gamma,episodes):
    all_rewards=[]
    labels=[]
    for eps in epsilons:
        Q,rewards=q_learning_frozenlake(env,alpha,gamma,eps,episodes)
        all_rewards.append(rewards)
        labels.append(f"epsilon={eps}")
    for rewards,label in zip(all_rewards,labels):
        plt.plot(rewards,label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning on FrozenLake for different epsilons")
    plt.legend()
    plt.show()
    return all_rewards,labels

def q_learning_with_policy_logging(env,alpha,gamma,epsilon,episodes):
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
        while not(terminated or truncated):
            a=behavior_policy_epsilon_greedy(Q,s,epsilon,env)
            greedy_a=np.argmax(Q[s])
            behavior_actions.append(a)
            target_actions.append(greedy_a)
            s2,r,terminated,truncated,_=env.step(a)
            best_next=np.max(Q[s2])
            Q[s,a]+=alpha*(r+gamma*best_next-Q[s,a])
            s=s2
            total_reward+=r
        rewards.append(total_reward)
    behavior_actions=np.array(behavior_actions)
    target_actions=np.array(target_actions)
    mismatches=(behavior_actions!=target_actions).sum()
    total=len(behavior_actions)
    mismatch_ratio=mismatches/total if total>0 else 0.0
    return Q,rewards,behavior_actions,target_actions,mismatch_ratio

def compare_mc_vs_q_learning_blackjack_frozenlake(mc_episodes,q_episodes,alpha,gamma,epsilon):
    env_bj=create_blackjack_env()
    env_fl=create_frozenlake_env()
    Q_mc=off_policy_mc_weighted_blackjack(env_bj,mc_episodes,gamma=1.0)
    Q_q,rew_q=q_learning_frozenlake(env_fl,alpha,gamma,epsilon,q_episodes)
    return Q_mc,Q_q,rew_q

def extract_greedy_policy_from_Q_frozenlake(Q):
    return np.argmax(Q,axis=1)

def decode_frozenlake_policy(policy):
    mapping={0:"L",1:"D",2:"R",3:"U"}
    return np.array([mapping[a] for a in policy])

def print_frozenlake_policy_grid(Q):
    policy=extract_greedy_policy_from_Q_frozenlake(Q)
    decoded=decode_frozenlake_policy(policy)
    n=int(np.sqrt(len(decoded)))
    grid=decoded.reshape((n,n))
    for row in grid:
        print(" ".join(row))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def create_env():
    return gym.make("FrozenLake-v1",is_slippery=False)

def epsilon_greedy(Q,state,epsilon,env):
    if np.random.random()<epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

def q_learning(env,alpha,gamma,epsilon,episodes):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    rewards=[]
    for episode in range(episodes):
        state,_=env.reset()
        total_reward=0
        terminated=False
        truncated=False
        while not(terminated or truncated):
            action=epsilon_greedy(Q,state,epsilon,env)
            next_state,reward,terminated,truncated,_=env.step(action)
            best_next=np.max(Q[next_state])
            Q[state,action]+=alpha*(reward+gamma*best_next-Q[state,action])
            state=next_state
            total_reward+=reward
        rewards.append(total_reward)
    return Q,rewards

def sarsa(env,alpha,gamma,epsilon,episodes):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    rewards=[]
    for episode in range(episodes):
        state,_=env.reset()
        action=epsilon_greedy(Q,state,epsilon,env)
        total_reward=0
        terminated=False
        truncated=False
        while not(terminated or truncated):
            next_state,reward,terminated,truncated,_=env.step(action)
            next_action=epsilon_greedy(Q,next_state,epsilon,env)
            Q[state,action]+=alpha*(reward+gamma*Q[next_state,next_action]-Q[state,action])
            state=next_state
            action=next_action
            total_reward+=reward
        rewards.append(total_reward)
    return Q,rewards

def expected_sarsa(env,alpha,gamma,epsilon,episodes):
    n_states=env.observation_space.n
    n_actions=env.action_space.n
    Q=np.zeros((n_states,n_actions))
    rewards=[]
    for episode in range(episodes):
        state,_=env.reset()
        total_reward=0
        terminated=False
        truncated=False
        while not(terminated or truncated):
            action=epsilon_greedy(Q,state,epsilon,env)
            next_state,reward,terminated,truncated,_=env.step(action)
            greedy_action=np.argmax(Q[next_state])
            probs=np.ones(n_actions)*(epsilon/n_actions)
            probs[greedy_action]+=1.0-epsilon
            expected_q=np.dot(probs,Q[next_state])
            Q[state,action]+=alpha*(reward+gamma*expected_q-Q[state,action])
            state=next_state
            total_reward+=reward
        rewards.append(total_reward)
    return Q,rewards

def plot_rewards(reward_lists,labels,title):
    for r,l in zip(reward_lists,labels):
        plt.plot(r,label=l)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    if len(labels)>1:
        plt.legend()
    plt.show()

def tune_hyperparams(algorithm,env,alpha_values,gamma_values,epsilon_values,episodes,last_k=100):
    results=[]
    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                if algorithm=="q_learning":
                    _,rewards=q_learning(env,alpha,gamma,epsilon,episodes)
                elif algorithm=="sarsa":
                    _,rewards=sarsa(env,alpha,gamma,epsilon,episodes)
                elif algorithm=="expected_sarsa":
                    _,rewards=expected_sarsa(env,alpha,gamma,epsilon,episodes)
                else:
                    continue
                avg_return=np.mean(rewards[-last_k:])
                results.append({"algorithm":algorithm,"alpha":alpha,"gamma":gamma,"epsilon":epsilon,"avg_return":avg_return})
    return results

def compare_q_learning_sarsa(env,alpha,gamma,epsilon,episodes):
    Q_q,rewards_q=q_learning(env,alpha,gamma,epsilon,episodes)
    Q_s,rewards_s=sarsa(env,alpha,gamma,epsilon,episodes)
    plot_rewards([rewards_q,rewards_s],["Q-learning","SARSA"],"Q-Learning vs SARSA Rewards")
    return Q_q,rewards_q,Q_s,rewards_s

def q_learning_decaying_epsilon(env,alpha,gamma,episodes,k=0.001):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    rewards=[]
    for episode in range(episodes):
        epsilon=1.0/(1.0+k*episode)
        state,_=env.reset()
        total_reward=0
        terminated=False
        truncated=False
        while not(terminated or truncated):
            action=epsilon_greedy(Q,state,epsilon,env)
            next_state,reward,terminated,truncated,_=env.step(action)
            best_next=np.max(Q[next_state])
            Q[state,action]+=alpha*(reward+gamma*best_next-Q[state,action])
            state=next_state
            total_reward+=reward
        rewards.append(total_reward)
    return Q,rewards

def sarsa_decaying_epsilon(env,alpha,gamma,episodes,k=0.001):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    rewards=[]
    for episode in range(episodes):
        epsilon=1.0/(1.0+k*episode)
        state,_=env.reset()
        action=epsilon_greedy(Q,state,epsilon,env)
        total_reward=0
        terminated=False
        truncated=False
        while not(terminated or truncated):
            next_state,reward,terminated,truncated,_=env.step(action)
            next_action=epsilon_greedy(Q,next_state,epsilon,env)
            Q[state,action]+=alpha*(reward+gamma*Q[next_state,next_action]-Q[state,action])
            state=next_state
            action=next_action
            total_reward+=reward
        rewards.append(total_reward)
    return Q,rewards

def extract_policy(Q):
    return np.argmax(Q,axis=1)

def decode_actions(policy):
    mapping={0:"L",1:"D",2:"R",3:"U"}
    return np.array([mapping[a] for a in policy])

def print_policy_grid(Q,env):
    policy=extract_policy(Q)
    decoded=decode_actions(policy)
    n=int(np.sqrt(env.observation_space.n))
    grid=decoded.reshape((n,n))
    for row in grid:
        print(" ".join(row))

def run_full_experiment():
    env=create_env()
    alpha=0.8
    gamma=0.95
    epsilon=0.1
    episodes=2000
    Q_q,rewards_q=q_learning(env,alpha,gamma,epsilon,episodes)
    Q_s,rewards_s=sarsa(env,alpha,gamma,epsilon,episodes)
    Q_e,rewards_e=expected_sarsa(env,alpha,gamma,epsilon,episodes)
    plot_rewards([rewards_q],["Q-learning"],"Q-Learning Rewards")
    plot_rewards([rewards_s],["SARSA"],"SARSA Rewards")
    plot_rewards([rewards_q,rewards_s,rewards_e],["Q-learning","SARSA","Expected SARSA"],"Algorithm Comparison")
    print("Q-learning policy:")
    print_policy_grid(Q_q,env)
    print("SARSA policy:")
    print_policy_grid(Q_s,env)
    print("Expected SARSA policy:")
    print_policy_grid(Q_e,env)

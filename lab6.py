import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. ENVIRONMENT SETUP =====================

def make_taxi_env(render_mode='ansi'):
    """
    Create Taxi-v3 environment.
    """
    env=gym.make('Taxi-v3',render_mode=render_mode)
    return env

def describe_taxi_env(env):
    """
    Print basic info about Taxi-v3.
    """
    print("Observation space (nS):",env.observation_space.n)
    print("Action space (nA):",env.action_space.n)
    print("Actions: 0=South,1=North,2=East,3=West,4=Pickup,5=Dropoff")
    print("Rewards: +20 for success, -1 per step, -10 illegal pickup/dropoff.")


# ===================== 2. VALUE ITERATION =====================

def value_iteration_taxi(env,discount_factor=0.99,theta=1e-6,
                         max_iterations=10000,return_deltas=False,verbose=False):
    """
    Value Iteration for Taxi-v3.
    Returns:
      V_opt, policy_opt, num_iterations, (optional) deltas_per_iter
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P
    V=np.zeros(nS)
    deltas=[]

    for i in range(max_iterations):
        delta=0
        V_new=np.zeros_like(V)
        for s in range(nS):
            q_sa=np.zeros(nA)
            for a in range(nA):
                for prob,next_state,reward,done in P[s][a]:
                    q_sa[a]+=prob*(reward+discount_factor*V[next_state])
            new_v=np.max(q_sa)
            delta=max(delta,abs(new_v-V[s]))
            V_new[s]=new_v
        V=V_new
        deltas.append(delta)
        if verbose:
            print(f"Iter {i+1}, delta={delta:.6f}")
        if delta<theta:
            break

    policy=extract_policy_from_v_taxi(env,V,discount_factor)
    if return_deltas:
        return V,policy,i+1,deltas
    return V,policy,i+1


# ===================== 3. POLICY FROM VALUE FUNCTION =====================

def extract_policy_from_v_taxi(env,V,discount_factor=0.99):
    """
    Extract greedy deterministic policy from value function V.
    """
    nS=env.observation_space.n
    nA=env.action_space.n
    P=env.unwrapped.P
    policy=np.zeros((nS,nA))

    for s in range(nS):
        q_sa=np.zeros(nA)
        for a in range(nA):
            for prob,next_state,reward,done in P[s][a]:
                q_sa[a]+=prob*(reward+discount_factor*V[next_state])
        best_a=int(np.argmax(q_sa))
        policy[s]=np.eye(nA)[best_a]
    return policy


# ===================== 4. VISUALIZATION FUNCTIONS =====================

def plot_taxi_values(V,title="Value Function for Taxi-v3"):
    """
    Plot V(s) for all 500 states.
    """
    plt.figure(figsize=(10,4))
    plt.plot(V)
    plt.title(title)
    plt.xlabel("State (0–499)")
    plt.ylabel("Value V(s)")
    plt.grid(True)
    plt.show()

def plot_taxi_policy(policy,title="Greedy Policy (Best Action per State)"):
    """
    Plot greedy action per state as bar chart.
    """
    nS=policy.shape[0]
    actions=np.argmax(policy,axis=1)
    plt.figure(figsize=(10,4))
    plt.bar(np.arange(nS),actions)
    plt.title(title)
    plt.xlabel("State index")
    plt.ylabel("Action (0=South,1=North,2=East,3=West,4=Pickup,5=Dropoff)")
    plt.show()

def plot_delta_convergence(deltas,title="Delta per Iteration (Value Iteration)"):
    """
    Plot max value change (delta) over iterations.
    """
    plt.figure()
    plt.plot(deltas)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Delta (max |V_new - V|)")
    plt.yscale("log")
    plt.grid(True)
    plt.show()


# ===================== 5. POLICY EVALUATION (SUCCESS RATE + AVG STEPS) =====================

def evaluate_policy_taxi(env,policy,n_episodes=1000,discount_factor=0.99):
    """
    Evaluate a deterministic policy on Taxi-v3.
    Returns:
      success_rate, avg_steps_success, avg_return
    success_rate: fraction of episodes with positive terminal reward
    avg_steps_success: avg steps over successful episodes only
    avg_return: average discounted return
    """
    successes=0
    total_steps_success=0
    returns=[]
    for _ in range(n_episodes):
        obs,_=env.reset()
        done=False
        steps=0
        G=0.0
        gamma_pow=1.0
        while not done:
            action=int(np.argmax(policy[obs]))
            obs,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            G+=gamma_pow*reward
            gamma_pow*=discount_factor
            steps+=1
            if done:
                if reward>0:
                    successes+=1
                    total_steps_success+=steps
                returns.append(G)
                break

    success_rate=successes/n_episodes if n_episodes>0 else 0.0
    avg_steps_success=(total_steps_success/successes) if successes>0 else 0.0
    avg_return=float(np.mean(returns)) if len(returns)>0 else 0.0
    return success_rate,avg_steps_success,avg_return


# ===================== 6. EXPERIMENT: DIFFERENT GAMMAS =====================

def experiment_gamma_taxi(gammas=(0.6,0.9,0.99),theta=1e-6,
                          max_iterations=10000,n_eval_episodes=500):
    """
    Run value iteration for different gamma values and compare:
      - iterations to converge
      - success rate
      - avg steps per success
      - avg return
    Returns dict[g] with stats.
    """
    results={}
    for g in gammas:
        print("\n=== Running Value Iteration for gamma =",g,"===")
        env=make_taxi_env()
        V,policy,iters,deltas=value_iteration_taxi(env,discount_factor=g,
                                                   theta=theta,
                                                   max_iterations=max_iterations,
                                                   return_deltas=True,
                                                   verbose=False)
        sr,avg_steps,avg_ret=evaluate_policy_taxi(env,policy,
                                                  n_episodes=n_eval_episodes,
                                                  discount_factor=g)
        results[g]={
            "V":V,
            "policy":policy,
            "iterations":iters,
            "deltas":deltas,
            "success_rate":sr,
            "avg_steps_success":avg_steps,
            "avg_return":avg_ret
        }
        print(f"Gamma={g}: iters={iters}, success={sr*100:.2f}%, avg_steps_success={avg_steps:.2f}, avg_return={avg_ret:.2f}")
    return results


# ===================== 7. MAIN DEMO (MIRRORS LAB STEPS) =====================

if __name__=="__main__":
    # Step 1 & 2: create env + describe
    env=make_taxi_env()
    describe_taxi_env(env)

    # Step 3: run value iteration
    gamma=0.99
    V_opt,policy_opt,iters,deltas=value_iteration_taxi(env,discount_factor=gamma,
                                                       theta=1e-6,
                                                       max_iterations=10000,
                                                       return_deltas=True,
                                                       verbose=True)
    print(f"\nConverged in {iters} iterations (gamma={gamma}).")

    # Step 5: plots
    plot_taxi_values(V_opt)
    plot_taxi_policy(policy_opt)
    plot_delta_convergence(deltas)

    # Optional Task 4: evaluate policy
    success_rate,avg_steps_success,avg_return=evaluate_policy_taxi(env,policy_opt,
                                                                   n_episodes=1000,
                                                                   discount_factor=gamma)
    print(f"\nSuccess Rate: {success_rate*100:.2f}%")
    print(f"Average steps per successful episode: {avg_steps_success:.2f}")
    print(f"Average discounted return: {avg_return:.2f}")

    # Optional: compare multiple gammas
    # results=experiment_gamma_taxi(gammas=(0.6,0.9,0.99))

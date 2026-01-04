import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def make_random_policy(seed=0):
    """Creates a random policy function policy(state,action_space)->action."""
    rng=np.random.default_rng(seed)
    def policy(state,action_space):
        return int(action_space.sample())
    return policy


def compute_returns_from_episode_states_rewards(episode,gamma):
    """Computes discounted returns G for each time step given episode as [(state,reward),...]."""
    G=0.0
    returns=[]
    for s,r in reversed(episode):
        G=float(r)+float(gamma)*G
        returns.append((int(s),float(G)))
    returns.reverse()
    return returns


def batch_mc_value_estimation(episodes,num_states,gamma=0.9):
    """Estimates V using batch Monte Carlo evaluation from fixed episodes."""
    V=np.zeros(int(num_states),dtype=np.float64)
    returns={int(s):[] for s in range(int(num_states))}
    for episode in episodes:
        Gs=compute_returns_from_episode_states_rewards(episode,gamma)
        for s,G in Gs:
            returns[int(s)].append(float(G))
    for s in range(int(num_states)):
        if len(returns[int(s)])>0:
            V[int(s)]=float(np.mean(np.array(returns[int(s)],dtype=np.float64)))
    return V,returns


def batch_td0_value_estimation(transitions,num_states,gamma=0.9,alpha=0.1,epochs=50):
    """Estimates V using batch TD(0) with synchronous updates over fixed transitions."""
    V=np.zeros(int(num_states),dtype=np.float64)
    history=[]
    for _ in range(int(epochs)):
        td_sums=np.zeros(int(num_states),dtype=np.float64)
        td_counts=np.zeros(int(num_states),dtype=np.float64)
        for s,r,s_next in transitions:
            s=int(s)
            s_next=int(s_next)
            td_error=float(r)+float(gamma)*float(V[s_next])-float(V[s])
            td_sums[s]+=td_error
            td_counts[s]+=1.0
        for s in range(int(num_states)):
            if td_counts[s]>0.0:
                V[s]+=float(alpha)*(td_sums[s]/td_counts[s])
        history.append(V.copy())
    return V,np.array(history,dtype=np.float64)


def batch_td_lambda_value_estimation(transitions,num_states,gamma=0.9,alpha=0.1,lam=0.8,epochs=50):
    """Estimates V using batch TD(lambda) with eligibility traces reset each epoch."""
    V=np.zeros(int(num_states),dtype=np.float64)
    history=[]
    for _ in range(int(epochs)):
        eligibility=np.zeros(int(num_states),dtype=np.float64)
        for s,r,s_next in transitions:
            s=int(s)
            s_next=int(s_next)
            td_error=float(r)+float(gamma)*float(V[s_next])-float(V[s])
            eligibility[s]+=1.0
            V=V+float(alpha)*td_error*eligibility
            eligibility=eligibility*(float(gamma)*float(lam))
        history.append(V.copy())
    return V,np.array(history,dtype=np.float64)


def lstd_value_estimation(transitions,num_states,gamma=0.9,reg=1e-6):
    """Computes V using LSTD with one-hot linear features for discrete states."""
    n=int(num_states)
    A=np.eye(n,dtype=np.float64)*float(reg)
    b=np.zeros(n,dtype=np.float64)
    for s,r,s_next in transitions:
        s=int(s)
        s_next=int(s_next)
        phi=np.zeros(n,dtype=np.float64)
        phi[s]=1.0
        phi_next=np.zeros(n,dtype=np.float64)
        phi_next[s_next]=1.0
        A+=np.outer(phi,phi-float(gamma)*phi_next)
        b+=phi*float(r)
    w=np.linalg.solve(A,b)
    V=w.copy()
    return V,w


def mse_to_reference(V,ref):
    """Computes MSE between V and reference array."""
    V=np.asarray(V,dtype=np.float64)
    ref=np.asarray(ref,dtype=np.float64)
    return float(np.mean((V-ref)**2))


def plot_value_history(history,labels,title):
    """Plots value components over epochs for multiple histories."""
    for H,label in zip(history,labels):
        H=np.asarray(H,dtype=np.float64)
        for s in range(H.shape[1]):
            plt.plot(H[:,s],label=f"{label}-V{s}")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_convergence_curve(histories,labels,title):
    """Plots convergence using mean absolute change across epochs."""
    for H,label in zip(histories,labels):
        H=np.asarray(H,dtype=np.float64)
        diffs=np.mean(np.abs(H[1:]-H[:-1]),axis=1) if H.shape[0]>1 else np.array([],dtype=np.float64)
        plt.plot(diffs,label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Mean |ΔV|")
    plt.title(title)
    plt.legend()
    plt.show()


def gamma_sweep_td(transitions,num_states,gammas=(0.5,0.9,0.99),alpha=0.1,epochs=50):
    """Runs gamma sweep for TD(0) and returns results."""
    out={}
    for g in gammas:
        V,H=batch_td0_value_estimation(transitions,num_states,gamma=float(g),alpha=float(alpha),epochs=int(epochs))
        out[float(g)]={"V":V,"history":H}
    return out


def lambda_sweep_td_lambda(transitions,num_states,lams=(0.0,0.5,0.8,1.0),gamma=0.9,alpha=0.1,epochs=50):
    """Runs lambda sweep for TD(lambda) and returns results."""
    out={}
    for lam in lams:
        V,H=batch_td_lambda_value_estimation(transitions,num_states,gamma=float(gamma),alpha=float(alpha),lam=float(lam),epochs=int(epochs))
        out[float(lam)]={"V":V,"history":H}
    return out


def epochs_sweep_td0(transitions,num_states,epoch_list=(10,25,50,100),gamma=0.9,alpha=0.1):
    """Runs TD(0) for different epochs and returns values."""
    out={}
    for e in epoch_list:
        V,H=batch_td0_value_estimation(transitions,num_states,gamma=float(gamma),alpha=float(alpha),epochs=int(e))
        out[int(e)]={"V":V,"history":H}
    return out


def collect_batch_transitions_from_env(env_id="FrozenLake-v1",episodes=2000,gamma=0.99,seed=0,max_steps=200,is_slippery=False,map_name="4x4"):
    """Collects a fixed dataset of (s,a,r,s_next,done) transitions under random policy."""
    env=gym.make(env_id,map_name=map_name,is_slippery=is_slippery) if env_id=="FrozenLake-v1" else gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    rng=np.random.default_rng(seed)
    transitions=[]
    for ep in range(int(episodes)):
        s,_=env.reset(seed=seed+ep)
        for _ in range(int(max_steps)):
            a=int(env.action_space.sample())
            s2,r,terminated,truncated,_=env.step(a)
            done=terminated or truncated
            transitions.append((int(s),int(a),float(r),int(s2),1 if done else 0))
            s=s2
            if done:
                break
    env.close()
    return transitions


def batch_td0_value_from_env_dataset(dataset,num_states,gamma=0.99,alpha=0.1,epochs=50):
    """Runs batch TD(0) value estimation from dataset of (s,a,r,s_next,done) using synchronous updates."""
    V=np.zeros(int(num_states),dtype=np.float64)
    history=[]
    for _ in range(int(epochs)):
        td_sums=np.zeros(int(num_states),dtype=np.float64)
        td_counts=np.zeros(int(num_states),dtype=np.float64)
        for s,a,r,s_next,done in dataset:
            s=int(s)
            s_next=int(s_next)
            target=float(r)+float(gamma)*(0.0 if int(done)==1 else float(V[s_next]))
            td_error=target-float(V[s])
            td_sums[s]+=td_error
            td_counts[s]+=1.0
        for s in range(int(num_states)):
            if td_counts[s]>0.0:
                V[s]+=float(alpha)*(td_sums[s]/td_counts[s])
        history.append(V.copy())
    return V,np.array(history,dtype=np.float64)


def batch_mc_value_from_env_episodes(env_id="FrozenLake-v1",episodes=2000,gamma=0.99,seed=0,max_steps=200,is_slippery=False,map_name="4x4"):
    """Collects full episodes and computes batch MC V from them."""
    env=gym.make(env_id,map_name=map_name,is_slippery=is_slippery) if env_id=="FrozenLake-v1" else gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    returns={}
    V=np.zeros(env.observation_space.n,dtype=np.float64)
    for s in range(env.observation_space.n):
        returns[int(s)]=[]
    for ep in range(int(episodes)):
        s,_=env.reset(seed=seed+ep)
        episode=[]
        for _ in range(int(max_steps)):
            a=int(env.action_space.sample())
            s2,r,terminated,truncated,_=env.step(a)
            episode.append((int(s),float(r)))
            s=s2
            if terminated or truncated:
                break
        Gs=compute_returns_from_episode_states_rewards(episode,gamma)
        for st,G in Gs:
            returns[int(st)].append(float(G))
    for s in range(env.observation_space.n):
        if len(returns[int(s)])>0:
            V[int(s)]=float(np.mean(np.array(returns[int(s)],dtype=np.float64)))
    env.close()
    return V,returns


def plot_mse_histories(histories,labels,reference,title):
    """Plots MSE across epochs relative to a reference V."""
    for H,label in zip(histories,labels):
        H=np.asarray(H,dtype=np.float64)
        mses=[]
        for t in range(H.shape[0]):
            mses.append(mse_to_reference(H[t],reference))
        plt.plot(mses,label=label)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.show()


def student_tasks_batch_methods():
    """Runs batch MC, batch TD(0), batch TD(lambda), and LSTD on fixed synthetic datasets plus env dataset."""
    episodes_data=[
        [(0,1),(1,1),(2,0)],
        [(0,1),(1,0),(2,1)],
        [(0,0),(1,1),(2,1)]
    ]
    num_states=3
    gamma=0.9

    V_mc,returns=batch_mc_value_estimation(episodes_data,num_states,gamma=gamma)

    transitions_data=[
        (0,1,1),
        (1,1,2),
        (2,0,2),
        (0,1,1),
        (1,0,2)
    ]
    V_td0,H_td0=batch_td0_value_estimation(transitions_data,num_states,gamma=gamma,alpha=0.1,epochs=50)
    V_tdlam,H_tdlam=batch_td_lambda_value_estimation([(0,1,1),(1,1,2),(2,0,2)],num_states,gamma=gamma,alpha=0.1,lam=0.8,epochs=50)
    V_lstd,w=lstd_value_estimation([(0,1,1),(1,1,2),(2,0,2)],num_states,gamma=gamma,reg=1e-6)

    plot_value_history([H_td0,H_tdlam],["TD0","TDlam"],"Batch Value Evolution (Synthetic)")
    plot_convergence_curve([H_td0,H_tdlam],["TD(0)","TD(lambda)"],"Convergence (Synthetic)")
    plot_mse_histories([H_td0,H_tdlam],["TD(0)","TD(lambda)"],V_mc,"MSE vs Batch MC Reference (Synthetic)")

    env_id="FrozenLake-v1"
    env_gamma=0.99
    env_seed=0
    env_episodes=3000
    env_is_slippery=False
    env_map="4x4"

    V_mc_env,_=batch_mc_value_from_env_episodes(env_id=env_id,episodes=env_episodes,gamma=env_gamma,seed=env_seed,max_steps=200,is_slippery=env_is_slippery,map_name=env_map)
    dataset=collect_batch_transitions_from_env(env_id=env_id,episodes=env_episodes,seed=env_seed,max_steps=200,is_slippery=env_is_slippery,map_name=env_map)
    num_states_env=16 if env_map=="4x4" else 64
    V_td0_env,H_td0_env=batch_td0_value_from_env_dataset(dataset,num_states_env,gamma=env_gamma,alpha=0.1,epochs=60)

    plot_convergence_curve([H_td0_env],["Batch TD(0)"],"Convergence (FrozenLake Dataset)")
    plot_mse_histories([H_td0_env],["Batch TD(0)"],V_mc_env,"MSE vs Batch MC Reference (FrozenLake Dataset)")

    return {
        "V_mc_synth":V_mc,
        "V_td0_synth":V_td0,
        "V_tdlam_synth":V_tdlam,
        "V_lstd_synth":V_lstd,
        "V_mc_env":V_mc_env,
        "V_td0_env":V_td0_env
    }


if __name__=="__main__":
    out=student_tasks_batch_methods()
    print(out["V_mc_synth"])
    print(out["V_td0_synth"])
    print(out["V_tdlam_synth"])
    print(out["V_lstd_synth"])

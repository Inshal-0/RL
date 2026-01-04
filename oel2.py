import math
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def make_frozenlake_env(seed=0,map_name="4x4",is_slippery=True,render_mode=None):
    """Creates FrozenLake-v1 environment with selected map size and stochasticity."""
    env=gym.make("FrozenLake-v1",map_name=map_name,is_slippery=is_slippery,render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_taxi_env(seed=0,render_mode=None):
    """Creates Taxi-v3 environment."""
    env=gym.make("Taxi-v3",render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class StepPenaltyWrapper(gym.Wrapper):
    def __init__(self,env,step_penalty=-0.01):
        super().__init__(env)
        self.step_penalty=float(step_penalty)

    def reset(self,**kwargs):
        return self.env.reset(**kwargs)

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        reward=float(reward)+self.step_penalty
        return obs,reward,terminated,truncated,info


class ActionNoiseWrapper(gym.Wrapper):
    def __init__(self,env,noise_prob=0.1,seed=0):
        super().__init__(env)
        self.noise_prob=float(noise_prob)
        self.rng=np.random.default_rng(seed)

    def reset(self,**kwargs):
        return self.env.reset(**kwargs)

    def step(self,action):
        if self.rng.random()<self.noise_prob:
            action=int(self.env.action_space.sample())
        return self.env.step(action)


def epsilon_schedule_constant(epsilon):
    """Creates a constant epsilon schedule function."""
    def f(t):
        return float(epsilon)
    return f


def epsilon_schedule_linear(eps_start,eps_end,decay_steps):
    """Creates a linear epsilon schedule function."""
    def f(t):
        if t>=decay_steps:
            return float(eps_end)
        frac=t/max(1,decay_steps)
        return float(eps_start+(eps_end-eps_start)*frac)
    return f


def epsilon_schedule_exponential(eps_start,eps_end,decay_rate):
    """Creates an exponential epsilon schedule function."""
    def f(t):
        val=eps_end+(eps_start-eps_end)*math.exp(-decay_rate*float(t))
        return float(max(eps_end,min(eps_start,val)))
    return f


def select_action_epsilon_greedy(Q,s,epsilon,rng,env):
    """Selects an action using epsilon-greedy for tabular Q."""
    if rng.random()<epsilon:
        return int(env.action_space.sample())
    return int(np.argmax(Q[s]))


def select_action_softmax(Q,s,tau,rng):
    """Selects an action using softmax exploration for tabular Q."""
    q=Q[s].astype(np.float64)
    z=(q-np.max(q))/max(1e-8,float(tau))
    p=np.exp(z)
    p=p/np.sum(p)
    return int(rng.choice(len(q),p=p))


def select_action_ucb(Q,N_s,step,c,rng):
    """Selects an action using UCB exploration for tabular Q."""
    n=np.maximum(1.0,N_s.astype(np.float64))
    bonus=float(c)*np.sqrt(np.log(max(2,int(step)))/n)
    a=int(np.argmax(Q+bonus))
    return a


def run_episode_env(env,policy,max_steps=1000,seed=None):
    """Runs one episode and returns (states,actions,rewards)."""
    s,_=env.reset(seed=seed)
    states=[]
    actions=[]
    rewards=[]
    for _ in range(max_steps):
        a=int(policy(s))
        s2,r,terminated,truncated,_=env.step(a)
        states.append(int(s))
        actions.append(int(a))
        rewards.append(float(r))
        s=s2
        if terminated or truncated:
            break
    return states,actions,rewards


def mc_prediction(env,policy,episodes=2000,gamma=0.99,seed=0,max_steps=1000):
    """Monte Carlo prediction (every-visit incremental mean) returning V history."""
    rng=np.random.default_rng(seed)
    nS=env.observation_space.n
    V=np.zeros(nS,dtype=np.float64)
    N=np.zeros(nS,dtype=np.float64)
    V_hist=[]
    for ep in range(episodes):
        states,actions,rewards=run_episode_env(env,policy,max_steps=max_steps,seed=seed+ep)
        G=0.0
        for t in range(len(states)-1,-1,-1):
            s=states[t]
            G=float(rewards[t])+gamma*G
            N[s]+=1.0
            V[s]+= (G-V[s])/N[s]
        V_hist.append(V.copy())
    return np.array(V_hist,dtype=np.float64)


def td0_prediction(env,policy,episodes=2000,alpha=0.1,gamma=0.99,seed=0,max_steps=1000):
    """TD(0) prediction returning V history."""
    nS=env.observation_space.n
    V=np.zeros(nS,dtype=np.float64)
    V_hist=[]
    for ep in range(episodes):
        s,_=env.reset(seed=seed+ep)
        for _ in range(max_steps):
            a=int(policy(s))
            s2,r,terminated,truncated,_=env.step(a)
            done=terminated or truncated
            target=float(r)+gamma*(0.0 if done else float(V[int(s2)]))
            V[int(s)]+=alpha*(target-float(V[int(s)]))
            s=s2
            if done:
                break
        V_hist.append(V.copy())
    return np.array(V_hist,dtype=np.float64)


def bellman_update_expected_v(env,V,policy,gamma=0.99):
    """Performs one Bellman expectation backup for V under a deterministic policy."""
    nS=env.observation_space.n
    V2=np.zeros(nS,dtype=np.float64)
    P=getattr(env.unwrapped,"P",None)
    if P is None:
        raise ValueError("Environment does not expose transition dictionary P for Bellman backup.")
    for s in range(nS):
        a=int(policy(s))
        val=0.0
        for prob,s2,r,done in P[s][a]:
            val+=float(prob)*(float(r)+gamma*(0.0 if done else float(V[int(s2)])))
        V2[s]=val
    return V2


def value_convergence_curve(V_hist):
    """Computes a convergence curve using mean absolute change between successive value estimates."""
    if len(V_hist)<2:
        return np.array([],dtype=np.float64)
    diffs=[]
    for t in range(1,len(V_hist)):
        diffs.append(float(np.mean(np.abs(V_hist[t]-V_hist[t-1]))))
    return np.array(diffs,dtype=np.float64)


def plot_mc_vs_td(V_mc_hist,V_td_hist,title="MC vs TD(0) Prediction"):
    """Plots convergence behavior for MC and TD(0) using mean absolute change curves."""
    mc_curve=value_convergence_curve(V_mc_hist)
    td_curve=value_convergence_curve(V_td_hist)
    plt.plot(mc_curve,label="MC Prediction")
    plt.plot(td_curve,label="TD(0) Prediction")
    plt.xlabel("Episode")
    plt.ylabel("Mean |ΔV|")
    plt.title(title)
    plt.legend()
    plt.show()


def train_sarsa(env,episodes=3000,alpha=0.1,gamma=0.99,epsilon_schedule=None,seed=0,max_steps=1000,exploration="epsilon_greedy",tau=1.0,ucb_c=1.0,optimistic_init=0.0,track_visits=False):
    """Trains SARSA and returns (Q,reward_list,visit_counts)."""
    rng=np.random.default_rng(seed)
    nS=env.observation_space.n
    nA=env.action_space.n
    Q=np.full((nS,nA),float(optimistic_init),dtype=np.float64)
    visits=np.zeros(nS,dtype=np.int64)
    rewards=[]
    global_step=0
    if epsilon_schedule is None:
        epsilon_schedule=epsilon_schedule_constant(0.1)

    for ep in range(episodes):
        s,_=env.reset(seed=seed+ep)
        s=int(s)
        if track_visits:
            visits[s]+=1
        eps=float(epsilon_schedule(global_step))
        if exploration=="softmax":
            a=select_action_softmax(Q,s,tau,rng)
        elif exploration=="ucb":
            a=select_action_ucb(Q[s],np.ones(nA,dtype=np.float64),max(1,global_step),ucb_c,rng)
        else:
            a=select_action_epsilon_greedy(Q,s,eps,rng,env)
        total=0.0

        for _ in range(max_steps):
            s2,r,terminated,truncated,_=env.step(int(a))
            s2=int(s2)
            done=terminated or truncated
            total+=float(r)
            global_step+=1
            if track_visits:
                visits[s2]+=1
            eps=float(epsilon_schedule(global_step))
            if done:
                Q[s,a]+=alpha*(float(r)-Q[s,a])
                break
            if exploration=="softmax":
                a2=select_action_softmax(Q,s2,tau,rng)
            elif exploration=="ucb":
                a2=select_action_ucb(Q[s2],np.ones(nA,dtype=np.float64),max(1,global_step),ucb_c,rng)
            else:
                a2=select_action_epsilon_greedy(Q,s2,eps,rng,env)
            td_target=float(r)+gamma*Q[s2,a2]
            Q[s,a]+=alpha*(td_target-Q[s,a])
            s=s2
            a=a2

        rewards.append(total)

    return Q,rewards,visits


def train_q_learning(env,episodes=3000,alpha=0.1,gamma=0.99,epsilon_schedule=None,seed=0,max_steps=1000,exploration="epsilon_greedy",tau=1.0,ucb_c=1.0,optimistic_init=0.0,track_visits=False):
    """Trains Q-Learning and returns (Q,reward_list,visit_counts)."""
    rng=np.random.default_rng(seed)
    nS=env.observation_space.n
    nA=env.action_space.n
    Q=np.full((nS,nA),float(optimistic_init),dtype=np.float64)
    visits=np.zeros(nS,dtype=np.int64)
    rewards=[]
    global_step=0
    if epsilon_schedule is None:
        epsilon_schedule=epsilon_schedule_constant(0.1)

    N_sa=np.zeros((nS,nA),dtype=np.int64)

    for ep in range(episodes):
        s,_=env.reset(seed=seed+ep)
        s=int(s)
        if track_visits:
            visits[s]+=1
        total=0.0

        for _ in range(max_steps):
            eps=float(epsilon_schedule(global_step))
            if exploration=="softmax":
                a=select_action_softmax(Q,s,tau,rng)
            elif exploration=="ucb":
                a=select_action_ucb(Q[s],np.maximum(1.0,N_sa[s].astype(np.float64)),max(2,global_step+2),ucb_c,rng)
            else:
                a=select_action_epsilon_greedy(Q,s,eps,rng,env)

            s2,r,terminated,truncated,_=env.step(int(a))
            s2=int(s2)
            done=terminated or truncated
            total+=float(r)
            global_step+=1
            N_sa[s,a]+=1
            if track_visits:
                visits[s2]+=1

            best_next=float(np.max(Q[s2]))
            td_target=float(r)+gamma*(0.0 if done else best_next)
            Q[s,a]+=alpha*(td_target-Q[s,a])

            s=s2
            if done:
                break

        rewards.append(total)

    return Q,rewards,visits


def extract_greedy_policy(Q):
    """Extracts greedy policy from Q-table."""
    return np.argmax(Q,axis=1).astype(np.int32)


def decode_frozenlake_policy(policy):
    """Decodes FrozenLake actions into symbols L,D,R,U."""
    mapping={0:"L",1:"D",2:"R",3:"U"}
    return np.array([mapping[int(a)] for a in policy])


def print_frozenlake_policy_grid(Q,map_name="4x4"):
    """Prints FrozenLake greedy policy as a grid."""
    policy=extract_greedy_policy(Q)
    decoded=decode_frozenlake_policy(policy)
    n=4 if map_name=="4x4" else 8 if map_name=="8x8" else int(np.sqrt(len(decoded)))
    grid=decoded.reshape((n,n))
    for row in grid:
        print(" ".join(row))


def plot_reward_curves(curves,labels,title="Reward Curves",window=50):
    """Plots reward curves with optional running mean."""
    for rewards,label in zip(curves,labels):
        r=np.asarray(rewards,dtype=np.float64)
        if window>1:
            rm=np.convolve(r,np.ones(window)/window,mode="valid")
            x=np.arange(len(rm))
            plt.plot(x,rm,label=label)
        else:
            plt.plot(r,label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.show()


def q_table_heatmap(Q,env_id="FrozenLake-v1",map_name="4x4",title="Q Heatmap (max_a Q)"):
    """Visualizes Q-table as a heatmap of max action-value per state for grid worlds."""
    vmax=np.max(Q,axis=1)
    if env_id=="FrozenLake-v1":
        n=4 if map_name=="4x4" else 8 if map_name=="8x8" else int(np.sqrt(len(vmax)))
        grid=vmax.reshape((n,n))
    else:
        grid=vmax.reshape((int(np.sqrt(len(vmax))),-1))
    plt.imshow(grid,aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.show()


def track_q_evolution_heatmaps(env_factory,trainer,episodes=2000,snapshots=(0.1,0.3,0.6,1.0),seed=0,title_prefix="Q Evolution"):
    """Trains with snapshots and plots Q heatmaps at training milestones."""
    snaps=[int(max(1,episodes*float(p))) for p in snapshots]
    snaps=sorted(list(set(snaps)))
    env=env_factory(seed=seed)
    nS=env.observation_space.n
    nA=env.action_space.n
    Q=np.zeros((nS,nA),dtype=np.float64)
    env.close()

    all_Q=[]
    done_at=set()
    cur=0
    for ep in range(episodes):
        cur=ep+1
        env=env_factory(seed=seed+ep)
        Q_tmp,rews,_=trainer(env,Q_seed=seed+ep)
        env.close()
        Q=Q_tmp
        if cur in snaps and cur not in done_at:
            all_Q.append((cur,Q.copy()))
            done_at.add(cur)

    for ep_count,Q_snap in all_Q:
        q_table_heatmap(Q_snap,env_id="FrozenLake-v1",map_name="4x4",title=f"{title_prefix} @ {ep_count} episodes")


def off_policy_mc_prediction_is_blackjack(episodes=20000,gamma=1.0,behavior="random",epsilon=0.9,seed=0):
    """Off-policy MC prediction for Blackjack using Ordinary and Weighted IS estimating V for target policy at initial states."""
    env=gym.make("Blackjack-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    rng=np.random.default_rng(seed)

    def target_policy(s):
        ps,ds,usable=s
        return 0 if ps>=20 else 1

    def behavior_policy(s):
        if behavior=="random":
            return int(rng.integers(0,2))
        if rng.random()<float(epsilon):
            return int(rng.integers(0,2))
        return int(target_policy(s))

    def b_prob(s,a):
        if behavior=="random":
            return 0.5
        if a==target_policy(s):
            return float((1.0-float(epsilon))+0.5*float(epsilon))
        return float(0.5*float(epsilon))

    V_ord=0.0
    V_w=0.0
    W_sum=0.0
    ord_hist=[]
    w_hist=[]
    for ep in range(episodes):
        s,_=env.reset(seed=seed+ep)
        done=False
        G=0.0
        W=1.0
        t=0
        while not done:
            a=behavior_policy(s)
            s2,r,terminated,truncated,_=env.step(int(a))
            done=terminated or truncated
            G=(float(gamma)**t)*float(r)+G
            pi_a=1.0 if int(a)==int(target_policy(s)) else 0.0
            b_a=float(b_prob(s,int(a)))
            if b_a<=0.0:
                W=0.0
            else:
                W*=float(pi_a/b_a)
            s=s2
            t+=1
            if W==0.0:
                break

        V_ord=V_ord+(W*G-V_ord)/float(ep+1)
        W_sum+=W
        if W_sum>0.0:
            V_w=V_w+(W/W_sum)*(G-V_w)
        ord_hist.append(float(V_ord))
        w_hist.append(float(V_w))

    env.close()
    return np.array(ord_hist,dtype=np.float64),np.array(w_hist,dtype=np.float64)


def plot_importance_sampling_histories(ord_hist,w_hist,title="Ordinary vs Weighted IS (Blackjack)"):
    """Plots IS estimate trajectories."""
    plt.plot(ord_hist,label="Ordinary IS")
    plt.plot(w_hist,label="Weighted IS")
    plt.xlabel("Episode")
    plt.ylabel("Estimated V")
    plt.title(title)
    plt.legend()
    plt.show()


def run_is_variance_experiment(episodes=20000,runs=10,behavior="random",epsilon=0.9,seed=0):
    """Runs multiple IS experiments and returns arrays of final estimates for variance comparison."""
    finals_ord=[]
    finals_w=[]
    for k in range(runs):
        ord_hist,w_hist=off_policy_mc_prediction_is_blackjack(episodes=episodes,behavior=behavior,epsilon=epsilon,seed=seed+1000*k)
        finals_ord.append(float(ord_hist[-1]))
        finals_w.append(float(w_hist[-1]))
    return np.array(finals_ord,dtype=np.float64),np.array(finals_w,dtype=np.float64)


def plot_is_variance(finals_ord,finals_w,title="IS Variance Comparison"):
    """Plots simple variance comparison bars."""
    vals=[float(np.var(finals_ord)),float(np.var(finals_w))]
    plt.bar(["Ordinary IS","Weighted IS"],vals)
    plt.ylabel("Variance of final estimate")
    plt.title(title)
    plt.show()


def reward_shaping_env(env,step_penalty=-0.01):
    """Wraps an environment with a small per-step penalty."""
    return StepPenaltyWrapper(env,step_penalty=step_penalty)


def compare_sarsa_qlearning_same_eps(env_factory,episodes=3000,alpha=0.1,gamma=0.99,epsilon_schedule=None,seed=0,max_steps=1000,step_penalty=0.0,track_visits=False):
    """Trains SARSA and Q-Learning on the same env settings and returns results."""
    env_s=env_factory(seed=seed)
    env_q=env_factory(seed=seed+999)
    if float(step_penalty)!=0.0:
        env_s=reward_shaping_env(env_s,step_penalty=step_penalty)
        env_q=reward_shaping_env(env_q,step_penalty=step_penalty)
    Qs,rs,vs=train_sarsa(env_s,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=epsilon_schedule,seed=seed,max_steps=max_steps,track_visits=track_visits)
    Qq,rq,vq=train_q_learning(env_q,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=epsilon_schedule,seed=seed,max_steps=max_steps,track_visits=track_visits)
    env_s.close()
    env_q.close()
    return (Qs,rs,vs),(Qq,rq,vq)


def compare_epsilon_schedules(env_factory,episodes=3000,alpha=0.1,gamma=0.99,seed=0,max_steps=1000):
    """Compares constant, linear, and exponential epsilon schedules for SARSA and Q-Learning."""
    schedules=[
        ("constant",epsilon_schedule_constant(0.1)),
        ("linear",epsilon_schedule_linear(1.0,0.05,int(episodes*0.8))),
        ("exponential",epsilon_schedule_exponential(1.0,0.05,0.002)),
    ]
    curves=[]
    labels=[]
    for name,sched in schedules:
        (Qs,rs,_),(Qq,rq,_)=compare_sarsa_qlearning_same_eps(env_factory,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=sched,seed=seed,max_steps=max_steps)
        curves.append(rs)
        labels.append(f"SARSA-{name}")
        curves.append(rq)
        labels.append(f"Q-{name}")
    plot_reward_curves(curves,labels,title="Epsilon Schedules: SARSA vs Q-Learning",window=50)


def experiment_mc_td_env_variations(seed=0):
    """Runs MC vs TD(0) prediction comparisons across environment variations and plots."""
    def pi_random(env,rng):
        def f(s):
            return int(env.action_space.sample())
        return f

    env1=make_frozenlake_env(seed=seed,map_name="4x4",is_slippery=False)
    pol1=pi_random(env1,np.random.default_rng(seed))
    Vmc1=mc_prediction(env1,pol1,episodes=2000,gamma=0.99,seed=seed,max_steps=200)
    Vtd1=td0_prediction(env1,pol1,episodes=2000,alpha=0.1,gamma=0.99,seed=seed,max_steps=200)
    env1.close()
    plot_mc_vs_td(Vmc1,Vtd1,title="FrozenLake 4x4 Non-Slippery: MC vs TD(0)")

    env2=make_frozenlake_env(seed=seed,map_name="4x4",is_slippery=True)
    pol2=pi_random(env2,np.random.default_rng(seed))
    Vmc2=mc_prediction(env2,pol2,episodes=2000,gamma=0.99,seed=seed,max_steps=200)
    Vtd2=td0_prediction(env2,pol2,episodes=2000,alpha=0.1,gamma=0.99,seed=seed,max_steps=200)
    env2.close()
    plot_mc_vs_td(Vmc2,Vtd2,title="FrozenLake 4x4 Slippery: MC vs TD(0)")

    env3=make_frozenlake_env(seed=seed,map_name="8x8",is_slippery=True)
    pol3=pi_random(env3,np.random.default_rng(seed))
    Vmc3=mc_prediction(env3,pol3,episodes=3000,gamma=0.99,seed=seed,max_steps=300)
    Vtd3=td0_prediction(env3,pol3,episodes=3000,alpha=0.1,gamma=0.99,seed=seed,max_steps=300)
    env3.close()
    plot_mc_vs_td(Vmc3,Vtd3,title="FrozenLake 8x8 Slippery: MC vs TD(0)")


def bellman_gamma_effect(env_factory,policy_factory,gammas=(0.5,0.9,0.99),iterations=50,seed=0):
    """Runs repeated Bellman expectation updates for different gammas and plots value heatmaps."""
    env=env_factory(seed=seed)
    pol=policy_factory(env)
    nS=env.observation_space.n
    for g in gammas:
        V=np.zeros(nS,dtype=np.float64)
        for _ in range(iterations):
            V=bellman_update_expected_v(env,V,pol,gamma=float(g))
        if hasattr(env.unwrapped,"desc"):
            desc=env.unwrapped.desc
            n=int(desc.shape[0])
            grid=V.reshape((n,n))
            plt.imshow(grid,aspect="auto")
            plt.colorbar()
            plt.title(f"Bellman V after {iterations} iters, gamma={g}")
            plt.show()
    env.close()


def q_learning_alpha_sweep(env_factory,alphas=(0.1,0.5,0.9),episodes=3000,gamma=0.99,seed=0,max_steps=1000):
    """Runs Q-Learning with different learning rates and plots reward curves."""
    curves=[]
    labels=[]
    for i,a in enumerate(alphas):
        env=env_factory(seed=seed+100*i)
        Q,r,_=train_q_learning(env,episodes=episodes,alpha=float(a),gamma=float(gamma),epsilon_schedule=epsilon_schedule_linear(1.0,0.05,int(episodes*0.8)),seed=seed+100*i,max_steps=max_steps)
        env.close()
        curves.append(r)
        labels.append(f"alpha={a}")
    plot_reward_curves(curves,labels,title="Q-Learning: Alpha Sweep",window=50)


def q_learning_gamma_sweep(env_factory,gammas=(0.5,0.9,0.99),episodes=3000,alpha=0.1,seed=0,max_steps=1000):
    """Runs Q-Learning with different discount factors and plots reward curves."""
    curves=[]
    labels=[]
    for i,g in enumerate(gammas):
        env=env_factory(seed=seed+200*i)
        Q,r,_=train_q_learning(env,episodes=episodes,alpha=float(alpha),gamma=float(g),epsilon_schedule=epsilon_schedule_linear(1.0,0.05,int(episodes*0.8)),seed=seed+200*i,max_steps=max_steps)
        env.close()
        curves.append(r)
        labels.append(f"gamma={g}")
    plot_reward_curves(curves,labels,title="Q-Learning: Gamma Sweep",window=50)


def compare_env_difficulty_sarsa_q(env_easy_factory,env_hard_factory,episodes=3000,alpha=0.1,gamma=0.99,seed=0,max_steps=1000):
    """Trains SARSA and Q-Learning on easy vs hard env and plots reward curves."""
    sched=epsilon_schedule_linear(1.0,0.05,int(episodes*0.8))
    (Qs_e,rs_e,_),(Qq_e,rq_e,_)=compare_sarsa_qlearning_same_eps(env_easy_factory,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=sched,seed=seed,max_steps=max_steps)
    (Qs_h,rs_h,_),(Qq_h,rq_h,_)=compare_sarsa_qlearning_same_eps(env_hard_factory,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=sched,seed=seed+9999,max_steps=max_steps)
    curves=[rs_e,rq_e,rs_h,rq_h]
    labels=["SARSA-easy","Q-easy","SARSA-hard","Q-hard"]
    plot_reward_curves(curves,labels,title="Degradation: Easy vs Hard",window=50)


def state_visitation_plot(visits,env_id="FrozenLake-v1",map_name="4x4",title="State Visitation Frequency"):
    """Plots state visitation frequency as a heatmap for grid worlds."""
    v=visits.astype(np.float64)
    if env_id=="FrozenLake-v1":
        n=4 if map_name=="4x4" else 8 if map_name=="8x8" else int(np.sqrt(len(v)))
        grid=v.reshape((n,n))
    else:
        grid=v.reshape((int(np.sqrt(len(v))),-1))
    plt.imshow(grid,aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.show()


def exploration_strategy_comparison(env_factory,episodes=3000,alpha=0.1,gamma=0.99,seed=0,max_steps=1000):
    """Compares epsilon-greedy, softmax, optimistic init, and UCB for Q-learning."""
    curves=[]
    labels=[]
    env=env_factory(seed=seed)
    Q1,r1,_=train_q_learning(env,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=epsilon_schedule_linear(1.0,0.05,int(episodes*0.8)),seed=seed,max_steps=max_steps,exploration="epsilon_greedy")
    env.close()
    curves.append(r1)
    labels.append("eps-greedy")

    env=env_factory(seed=seed+10)
    Q2,r2,_=train_q_learning(env,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=epsilon_schedule_constant(0.0),seed=seed+10,max_steps=max_steps,exploration="softmax",tau=0.8)
    env.close()
    curves.append(r2)
    labels.append("softmax")

    env=env_factory(seed=seed+20)
    Q3,r3,_=train_q_learning(env,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=epsilon_schedule_linear(1.0,0.05,int(episodes*0.8)),seed=seed+20,max_steps=max_steps,exploration="epsilon_greedy",optimistic_init=1.0)
    env.close()
    curves.append(r3)
    labels.append("optimistic")

    env=env_factory(seed=seed+30)
    Q4,r4,_=train_q_learning(env,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=epsilon_schedule_constant(0.0),seed=seed+30,max_steps=max_steps,exploration="ucb",ucb_c=1.5)
    env.close()
    curves.append(r4)
    labels.append("ucb")

    plot_reward_curves(curves,labels,title="Exploration Strategies (Q-Learning)",window=50)


def noise_instability_experiment(env_factory,noise_probs=(0.0,0.05,0.1,0.2),episodes=3000,alpha=0.1,gamma=0.99,seed=0,max_steps=1000):
    """Adds action noise and compares SARSA vs Q-Learning stability across noise levels."""
    curves=[]
    labels=[]
    sched=epsilon_schedule_linear(1.0,0.05,int(episodes*0.8))
    for i,p in enumerate(noise_probs):
        envS=env_factory(seed=seed+100*i)
        envQ=env_factory(seed=seed+100*i+50)
        envS=ActionNoiseWrapper(envS,noise_prob=float(p),seed=seed+100*i)
        envQ=ActionNoiseWrapper(envQ,noise_prob=float(p),seed=seed+100*i+1)
        Qs,rs,_=train_sarsa(envS,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=sched,seed=seed+100*i,max_steps=max_steps)
        Qq,rq,_=train_q_learning(envQ,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_schedule=sched,seed=seed+100*i+1,max_steps=max_steps)
        envS.close()
        envQ.close()
        curves.append(rs)
        labels.append(f"SARSA-noise={p}")
        curves.append(rq)
        labels.append(f"Q-noise={p}")
    plot_reward_curves(curves,labels,title="Noise Sensitivity: SARSA vs Q-Learning",window=50)


def student_tasks_lab02(seed=0):
    """Runs Lab 02 experiments end-to-end with plots."""
    experiment_mc_td_env_variations(seed=seed)

    def env_factory_fl_easy(seed=0):
        return make_frozenlake_env(seed=seed,map_name="4x4",is_slippery=False)

    def env_factory_fl_hard(seed=0):
        return make_frozenlake_env(seed=seed,map_name="8x8",is_slippery=True)

    compare_epsilon_schedules(env_factory_fl_easy,episodes=3000,alpha=0.1,gamma=0.99,seed=seed,max_steps=200)

    (Qs,rs,vs),(Qq,rq,vq)=compare_sarsa_qlearning_same_eps(env_factory_fl_easy,episodes=3000,alpha=0.1,gamma=0.99,epsilon_schedule=epsilon_schedule_linear(1.0,0.05,2400),seed=seed,max_steps=200,track_visits=True)
    plot_reward_curves([rs,rq],["SARSA","Q-Learning"],title="FrozenLake 4x4 Non-Slippery: SARSA vs Q",window=50)
    q_table_heatmap(Qs,env_id="FrozenLake-v1",map_name="4x4",title="SARSA Q Heatmap")
    q_table_heatmap(Qq,env_id="FrozenLake-v1",map_name="4x4",title="Q-Learning Q Heatmap")
    state_visitation_plot(vs,env_id="FrozenLake-v1",map_name="4x4",title="SARSA State Visitation")
    state_visitation_plot(vq,env_id="FrozenLake-v1",map_name="4x4",title="Q-Learning State Visitation")

    compare_env_difficulty_sarsa_q(env_factory_fl_easy,env_factory_fl_hard,episodes=3000,alpha=0.1,gamma=0.99,seed=seed,max_steps=300)

    (Qs_p,rs_p,_),(Qq_p,rq_p,_)=compare_sarsa_qlearning_same_eps(env_factory_fl_easy,episodes=3000,alpha=0.1,gamma=0.99,epsilon_schedule=epsilon_schedule_linear(1.0,0.05,2400),seed=seed,max_steps=200,step_penalty=-0.01)
    plot_reward_curves([rs_p,rq_p],["SARSA (step penalty)","Q (step penalty)"],title="Reward Shaping: Step Penalty",window=50)

    q_learning_alpha_sweep(env_factory_fl_easy,alphas=(0.1,0.5,0.9),episodes=3000,gamma=0.99,seed=seed,max_steps=200)
    q_learning_gamma_sweep(env_factory_fl_easy,gammas=(0.5,0.9,0.99),episodes=3000,alpha=0.1,seed=seed,max_steps=200)

    ord_hist,w_hist=off_policy_mc_prediction_is_blackjack(episodes=20000,behavior="random",epsilon=0.9,seed=seed)
    plot_importance_sampling_histories(ord_hist,w_hist,title="Blackjack: Ordinary vs Weighted IS (Random Behavior)")
    finals_ord,finals_w=run_is_variance_experiment(episodes=20000,runs=10,behavior="random",epsilon=0.9,seed=seed)
    plot_is_variance(finals_ord,finals_w,title="Blackjack: Variance (Random Behavior)")

    ord_hist2,w_hist2=off_policy_mc_prediction_is_blackjack(episodes=20000,behavior="epsilon_greedy",epsilon=0.9,seed=seed+1)
    plot_importance_sampling_histories(ord_hist2,w_hist2,title="Blackjack: Ordinary vs Weighted IS (High-epsilon Behavior)")
    finals_ord2,finals_w2=run_is_variance_experiment(episodes=20000,runs=10,behavior="epsilon_greedy",epsilon=0.9,seed=seed+1)
    plot_is_variance(finals_ord2,finals_w2,title="Blackjack: Variance (High-epsilon Behavior)")

    exploration_strategy_comparison(env_factory_fl_easy,episodes=3000,alpha=0.1,gamma=0.99,seed=seed,max_steps=200)
    noise_instability_experiment(env_factory_fl_easy,noise_probs=(0.0,0.05,0.1,0.2),episodes=3000,alpha=0.1,gamma=0.99,seed=seed,max_steps=200)

    return {"Qs":Qs,"Qq":Qq,"rs":rs,"rq":rq}


if __name__=="__main__":
    student_tasks_lab02(seed=0)

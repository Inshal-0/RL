import math
import random
from collections import deque

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


def make_cartpole_env(seed=0,render_mode=None):
    """Creates CartPole-v1 environment."""
    env=gym.make("CartPole-v1",render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class QNetwork(nn.Module):
    def __init__(self,state_dim,action_dim,hidden=64,activation="relu"):
        super().__init__()
        act=nn.ReLU if activation=="relu" else nn.Tanh
        self.fc1=nn.Linear(state_dim,hidden)
        self.fc2=nn.Linear(hidden,hidden)
        self.fc3=nn.Linear(hidden,action_dim)
        self.act=act()

    def forward(self,x):
        x=self.act(self.fc1(x))
        x=self.act(self.fc2(x))
        return self.fc3(x)


def to_tensor(x,device):
    """Converts numpy array to torch float tensor on device."""
    return torch.as_tensor(x,dtype=torch.float32,device=device)


def select_action_epsilon_greedy(env,q_net,state,epsilon,rng,device):
    """Selects an action using epsilon-greedy strategy."""
    if rng.random()<float(epsilon):
        return int(env.action_space.sample())
    s=to_tensor(np.asarray(state,dtype=np.float32)[None,:],device)
    with torch.no_grad():
        return int(torch.argmax(q_net(s),dim=1).item())


def compute_td_targets(target_net,next_states,rewards,dones,gamma,device):
    """Computes TD targets for a batch using target network."""
    with torch.no_grad():
        ns=to_tensor(next_states,device)
        max_next=target_net(ns).max(dim=1,keepdim=True).values
        r=to_tensor(rewards,device).unsqueeze(1)
        d=to_tensor(dones,device).unsqueeze(1)
        return r+float(gamma)*max_next*(1.0-d)


def sample_batch(memory,batch_size,rng):
    """Samples a mini-batch from replay memory."""
    batch=rng.sample(list(memory),int(batch_size))
    states,actions,rewards,next_states,dones=zip(*batch)
    return (
        np.asarray(states,dtype=np.float32),
        np.asarray(actions,dtype=np.int64),
        np.asarray(rewards,dtype=np.float32),
        np.asarray(next_states,dtype=np.float32),
        np.asarray(dones,dtype=np.float32),
    )


def dqn_replay_step(q_net,target_net,optimizer,memory,batch_size,gamma,device,loss_type="mse",rng=None):
    """Performs one batch replay update step."""
    if len(memory)<int(batch_size):
        return None
    if rng is None:
        rng=random.Random()
    states,actions,rewards,next_states,dones=sample_batch(memory,batch_size,rng)
    s=to_tensor(states,device)
    a=torch.as_tensor(actions,dtype=torch.int64,device=device).unsqueeze(1)
    q_values=q_net(s).gather(1,a)
    targets=compute_td_targets(target_net,next_states,rewards,dones,gamma,device)
    if loss_type=="huber":
        loss_fn=nn.SmoothL1Loss()
    else:
        loss_fn=nn.MSELoss()
    loss=loss_fn(q_values,targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(),10.0)
    optimizer.step()
    return float(loss.item())


def moving_average(x,window=10):
    """Computes moving average."""
    x=np.asarray(x,dtype=np.float64)
    if len(x)<int(window):
        return np.array([],dtype=np.float64)
    w=int(window)
    return np.convolve(x,np.ones(w)/w,mode="valid")


def first_episode_above_threshold(rewards,threshold=195,window=10):
    """Finds first episode index where moving average exceeds threshold."""
    ma=moving_average(rewards,window=window)
    if ma.size==0:
        return None
    idx=np.where(ma>=float(threshold))[0]
    if idx.size==0:
        return None
    return int(idx[0]+window-1)


def train_dqn(
    episodes=200,
    max_steps=500,
    gamma=0.99,
    lr=0.001,
    batch_size=64,
    memory_size=2000,
    target_update=10,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    hidden=64,
    activation="relu",
    loss_type="mse",
    seed=0,
    device="cuda"
):
    """Trains a DQN on CartPole-v1 using experience replay and target network."""
    if device=="cuda" and not torch.cuda.is_available():
        device="cpu"
    env=make_cartpole_env(seed=seed)
    rng_env=np.random.default_rng(seed)
    rng_py=random.Random(seed)

    state_dim=int(env.observation_space.shape[0])
    action_dim=int(env.action_space.n)

    q_net=QNetwork(state_dim,action_dim,hidden=hidden,activation=activation).to(device)
    target_net=QNetwork(state_dim,action_dim,hidden=hidden,activation=activation).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer=optim.Adam(q_net.parameters(),lr=float(lr))
    memory=deque(maxlen=int(memory_size))

    rewards_list=[]
    losses=[]
    epsilon=float(epsilon_start)

    for ep in range(int(episodes)):
        state,_=env.reset(seed=seed+ep)
        total_reward=0.0
        done=False
        steps=0
        while not done and steps<int(max_steps):
            action=select_action_epsilon_greedy(env,q_net,state,epsilon,rng_py,device)
            next_state,reward,terminated,truncated,_=env.step(int(action))
            done=terminated or truncated
            memory.append((np.asarray(state,dtype=np.float32),int(action),float(reward),np.asarray(next_state,dtype=np.float32),1.0 if done else 0.0))
            state=next_state
            total_reward+=float(reward)
            loss=dqn_replay_step(q_net,target_net,optimizer,memory,batch_size,gamma,device,loss_type=loss_type,rng=rng_py)
            if loss is not None:
                losses.append(loss)
            steps+=1

        rewards_list.append(total_reward)
        epsilon=max(float(epsilon_min),float(epsilon)*float(epsilon_decay))
        if int(target_update)>0 and (ep%int(target_update)==0):
            target_net.load_state_dict(q_net.state_dict())

    env.close()
    return q_net,target_net,rewards_list,losses


def plot_rewards(rewards,title="DQN Learning Curve"):
    """Plots episode rewards and moving average."""
    r=np.asarray(rewards,dtype=np.float64)
    plt.plot(r,label="Reward")
    ma=moving_average(r,window=10)
    if ma.size>0:
        plt.plot(np.arange(len(ma))+9,ma,label="MA(10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.show()


def run_activation_comparison(seed=0):
    """Runs ReLU vs Tanh comparison and plots curves."""
    _,_,r1,_=train_dqn(episodes=200,activation="relu",seed=seed,loss_type="mse")
    _,_,r2,_=train_dqn(episodes=200,activation="tanh",seed=seed+1,loss_type="mse")
    plot_rewards(r1,title="DQN ReLU")
    plot_rewards(r2,title="DQN Tanh")
    return r1,r2


def run_epsilon_decay_comparison(seed=0):
    """Runs epsilon_decay comparison and plots curves."""
    _,_,r1,_=train_dqn(episodes=200,epsilon_decay=0.995,seed=seed)
    _,_,r2,_=train_dqn(episodes=200,epsilon_decay=0.99,seed=seed+1)
    _,_,r3,_=train_dqn(episodes=200,epsilon_start=0.1,epsilon_decay=1.0,epsilon_min=0.1,seed=seed+2)
    plot_rewards(r1,title="epsilon_decay=0.995")
    plot_rewards(r2,title="epsilon_decay=0.99")
    plot_rewards(r3,title="epsilon fixed at 0.1")
    return r1,r2,r3


def run_batch_size_comparison(seed=0):
    """Runs batch size comparison and plots curves."""
    _,_,r1,_=train_dqn(episodes=200,batch_size=32,seed=seed)
    _,_,r2,_=train_dqn(episodes=200,batch_size=64,seed=seed+1)
    _,_,r3,_=train_dqn(episodes=200,batch_size=128,seed=seed+2)
    plot_rewards(r1,title="batch_size=32")
    plot_rewards(r2,title="batch_size=64")
    plot_rewards(r3,title="batch_size=128")
    return r1,r2,r3


def run_target_update_comparison(seed=0):
    """Runs target_update comparison and plots curves."""
    _,_,r1,_=train_dqn(episodes=200,target_update=1,seed=seed)
    _,_,r2,_=train_dqn(episodes=200,target_update=10,seed=seed+1)
    _,_,r3,_=train_dqn(episodes=200,target_update=50,seed=seed+2)
    plot_rewards(r1,title="target_update=1")
    plot_rewards(r2,title="target_update=10")
    plot_rewards(r3,title="target_update=50")
    return r1,r2,r3


def run_loss_function_comparison(seed=0):
    """Runs MSE vs Huber loss comparison and plots curves."""
    _,_,r1,l1=train_dqn(episodes=200,loss_type="mse",seed=seed)
    _,_,r2,l2=train_dqn(episodes=200,loss_type="huber",seed=seed+1)
    plot_rewards(r1,title="Loss=MSE")
    plot_rewards(r2,title="Loss=Huber")
    return r1,r2,l1,l2


def run_hyperparameter_sensitivity(seed=0):
    """Runs single-parameter sensitivity experiments and plots results."""
    configs=[
        ("lr=0.001",{"lr":0.001}),
        ("lr=0.0005",{"lr":0.0005}),
        ("gamma=0.99",{"gamma":0.99}),
        ("gamma=0.95",{"gamma":0.95}),
        ("batch=64",{"batch_size":64}),
        ("batch=128",{"batch_size":128}),
        ("episodes=200",{"episodes":200}),
        ("episodes=400",{"episodes":400})
    ]
    results={}
    for i,(name,kw) in enumerate(configs):
        _,_,r,_=train_dqn(seed=seed+10*i,**kw)
        results[name]=r
    for k,v in results.items():
        plot_rewards(v,title=f"DQN {k}")
    return results


def student_tasks_lab12(seed=0):
    """Runs Lab 12 default training, plots rewards, and reports solve episode."""
    _,_,rewards,_=train_dqn(episodes=200,seed=seed,batch_size=64,gamma=0.99,lr=0.001,target_update=10,epsilon_decay=0.995,loss_type="mse",activation="relu")
    plot_rewards(rewards,title="DQN Learning Curve (Default)")
    solved_at=first_episode_above_threshold(rewards,threshold=195,window=10)
    return rewards,solved_at


if __name__=="__main__":
    rewards,solved_at=student_tasks_lab12(seed=0)
    print(solved_at)

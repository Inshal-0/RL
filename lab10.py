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


def select_random_action(env,rng):
    """Selects a random action."""
    return int(env.action_space.sample())


def feature_vector_linear(s):
    """Generates linear feature vector [1,s]."""
    return np.array([1.0,float(s)],dtype=np.float64)


def feature_vector_poly2(s):
    """Generates polynomial feature vector [1,s,s^2]."""
    x=float(s)
    return np.array([1.0,x,x*x],dtype=np.float64)


def vhat_linear(w,s,feature_fn):
    """Computes approximate V(s)=w^T phi(s)."""
    phi=feature_fn(s)
    return float(np.dot(w,phi))


def td0_update_linear(w,s,r,s_next,done,alpha,gamma,feature_fn):
    """Performs TD(0) update for linear value function approximation."""
    phi=feature_fn(s)
    v_s=float(np.dot(w,phi))
    v_next=0.0 if done else float(np.dot(w,feature_fn(s_next)))
    td_error=float(r)+float(gamma)*v_next-v_s
    w=w+float(alpha)*td_error*phi
    return w,td_error


def run_td0_linear_value_approx(env,episodes=200,alpha=0.01,gamma=0.99,seed=0,max_steps=500,feature_fn=feature_vector_linear):
    """Runs TD(0) prediction with linear value approximation under random policy."""
    rng=np.random.default_rng(seed)
    w=np.zeros(int(len(feature_fn(0.0))),dtype=np.float64)
    ep_mse=[]
    ep_mean_abs_td=[]
    for ep in range(episodes):
        state,_=env.reset(seed=seed+ep)
        done=False
        td_errors=[]
        preds=[]
        targets=[]
        steps=0
        while not done and steps<max_steps:
            a=select_random_action(env,rng)
            next_state,reward,terminated,truncated,_=env.step(a)
            done=terminated or truncated
            s=float(state[0])
            s_next=float(next_state[0])
            v_before=vhat_linear(w,s,feature_fn)
            target=float(reward)+float(gamma)*(0.0 if done else vhat_linear(w,s_next,feature_fn))
            w,td_err=td0_update_linear(w,s,reward,s_next,done,alpha,gamma,feature_fn)
            td_errors.append(float(td_err))
            preds.append(float(v_before))
            targets.append(float(target))
            state=next_state
            steps+=1
        if len(td_errors)==0:
            ep_mean_abs_td.append(0.0)
            ep_mse.append(0.0)
        else:
            ep_mean_abs_td.append(float(np.mean(np.abs(np.array(td_errors,dtype=np.float64)))))
            p=np.array(preds,dtype=np.float64)
            t=np.array(targets,dtype=np.float64)
            ep_mse.append(float(np.mean((p-t)**2)))
    return w,np.array(ep_mse,dtype=np.float64),np.array(ep_mean_abs_td,dtype=np.float64)


class ValueNet(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(1,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )

    def forward(self,x):
        return self.fc(x)


def td_target_network(model,next_state_scalar,reward,gamma,done,device):
    """Computes TD target for neural value approximation."""
    if done:
        return torch.as_tensor([float(reward)],dtype=torch.float32,device=device)
    with torch.no_grad():
        ns=torch.as_tensor([[float(next_state_scalar)]],dtype=torch.float32,device=device)
        v_next=model(ns).squeeze(1)
        return torch.as_tensor([float(reward)],dtype=torch.float32,device=device)+float(gamma)*v_next


def run_td0_nn_value_approx(env,episodes=200,lr=0.001,gamma=0.99,seed=0,max_steps=500,hidden=64,device="cuda"):
    """Runs TD(0) prediction with neural network value approximation under random policy."""
    rng=np.random.default_rng(seed)
    if device=="cuda" and not torch.cuda.is_available():
        device="cpu"
    model=ValueNet(hidden=hidden).to(device)
    optimizer=optim.Adam(model.parameters(),lr=float(lr))
    loss_fn=nn.MSELoss()

    ep_loss=[]
    ep_mean_abs_td=[]

    for ep in range(episodes):
        state,_=env.reset(seed=seed+ep)
        done=False
        losses=[]
        td_abs=[]
        steps=0
        while not done and steps<max_steps:
            s=float(state[0])
            s_t=torch.as_tensor([[s]],dtype=torch.float32,device=device)
            value=model(s_t).squeeze(1)

            a=select_random_action(env,rng)
            next_state,reward,terminated,truncated,_=env.step(a)
            done=terminated or truncated
            s_next=float(next_state[0])

            target=td_target_network(model,s_next,reward,gamma,done,device)
            td_err=(target-value.detach()).abs().item()
            loss=loss_fn(value,target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            td_abs.append(float(td_err))
            state=next_state
            steps+=1

        ep_loss.append(float(np.mean(losses)) if len(losses)>0 else 0.0)
        ep_mean_abs_td.append(float(np.mean(td_abs)) if len(td_abs)>0 else 0.0)

    return model,np.array(ep_loss,dtype=np.float64),np.array(ep_mean_abs_td,dtype=np.float64)


def tabular_td0_prediction_discretized(env,episodes=200,alpha=0.1,gamma=0.99,seed=0,max_steps=500,bins=25,range_min=-2.4,range_max=2.4):
    """Runs tabular TD(0) prediction on discretized CartPole position dimension under random policy."""
    rng=np.random.default_rng(seed)
    edges=np.linspace(float(range_min),float(range_max),int(bins)+1)
    V=np.zeros(int(bins),dtype=np.float64)
    N=np.zeros(int(bins),dtype=np.float64)
    ep_mean_abs_td=[]
    for ep in range(episodes):
        state,_=env.reset(seed=seed+ep)
        done=False
        td_abs=[]
        steps=0
        while not done and steps<max_steps:
            x=float(state[0])
            b=int(np.clip(np.digitize(x,edges)-1,0,bins-1))
            a=select_random_action(env,rng)
            next_state,reward,terminated,truncated,_=env.step(a)
            done=terminated or truncated
            x2=float(next_state[0])
            b2=int(np.clip(np.digitize(x2,edges)-1,0,bins-1))
            target=float(reward)+float(gamma)*(0.0 if done else float(V[b2]))
            td_err=target-float(V[b])
            V[b]+=float(alpha)*td_err
            td_abs.append(abs(float(td_err)))
            state=next_state
            steps+=1
        ep_mean_abs_td.append(float(np.mean(td_abs)) if len(td_abs)>0 else 0.0)
    return V,np.array(ep_mean_abs_td,dtype=np.float64),edges


def approx_memory_bytes_tabular(bins):
    """Computes approximate memory in bytes for a tabular V with given bins."""
    return int(np.zeros(int(bins),dtype=np.float64).nbytes)


def approx_memory_bytes_linear(w):
    """Computes approximate memory in bytes for linear weights."""
    return int(np.asarray(w,dtype=np.float64).nbytes)


def approx_memory_bytes_network(model):
    """Computes approximate memory in bytes for a torch model's parameters."""
    total=0
    for p in model.parameters():
        total+=p.numel()*p.element_size()
    return int(total)


def plot_learning_curves(curves,labels,title,ylabel,window=10):
    """Plots learning curves with running mean."""
    for y,label in zip(curves,labels):
        y=np.asarray(y,dtype=np.float64)
        if window>1 and len(y)>=window:
            rm=np.convolve(y,np.ones(window)/window,mode="valid")
            plt.plot(rm,label=label)
        else:
            plt.plot(y,label=label)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def experiment_compare_tabular_linear_nn(seed=0,episodes=200):
    """Runs experiment comparing discretized tabular TD, linear TD, and NN TD on CartPole."""
    env=make_cartpole_env(seed=seed)

    V_tab,tab_td,edges=tabular_td0_prediction_discretized(env,episodes=episodes,alpha=0.1,gamma=0.99,seed=seed,max_steps=500,bins=25)
    w_lin,lin_mse,lin_td=run_td0_linear_value_approx(env,episodes=episodes,alpha=0.01,gamma=0.99,seed=seed,max_steps=500,feature_fn=feature_vector_linear)
    model,nn_loss,nn_td=run_td0_nn_value_approx(env,episodes=episodes,lr=0.001,gamma=0.99,seed=seed,max_steps=500,hidden=64,device="cuda")

    env.close()

    m_tab=approx_memory_bytes_tabular(25)
    m_lin=approx_memory_bytes_linear(w_lin)
    m_nn=approx_memory_bytes_network(model)

    plot_learning_curves([tab_td,lin_td,nn_td],["Tabular TD(0)","Linear TD(0)","NN TD(0)"],"Mean |TD Error| vs Episode","Mean |TD Error|",window=10)
    plot_learning_curves([lin_mse,nn_loss],["Linear MSE","NN Loss"],"Prediction Loss vs Episode","Loss",window=10)

    return {
        "tabular_memory_bytes":m_tab,
        "linear_memory_bytes":m_lin,
        "nn_memory_bytes":m_nn,
        "linear_weights":w_lin,
        "tabular_td_abs":tab_td,
        "linear_td_abs":lin_td,
        "nn_td_abs":nn_td,
        "linear_mse":lin_mse,
        "nn_loss":nn_loss
    }


def experiment_learning_rate_effect_linear(seed=0,episodes=200,alphas=(0.001,0.01,0.05),poly=False):
    """Runs experiment testing different learning rates for linear TD value approximation."""
    env=make_cartpole_env(seed=seed)
    curves=[]
    labels=[]
    for i,a in enumerate(alphas):
        feature_fn=feature_vector_poly2 if bool(poly) else feature_vector_linear
        w,ep_mse,ep_td=run_td0_linear_value_approx(env,episodes=episodes,alpha=float(a),gamma=0.99,seed=seed+100*i,max_steps=500,feature_fn=feature_fn)
        curves.append(ep_td)
        labels.append(f"alpha={a}")
    env.close()
    plot_learning_curves(curves,labels,"Linear TD(0): Mean |TD Error| for different alpha","Mean |TD Error|",window=10)
    return curves,labels


def student_tasks_lab10(seed=0):
    """Runs Lab 10 tasks and experiments with plots."""
    env=make_cartpole_env(seed=seed)
    w_lin,lin_mse,lin_td=run_td0_linear_value_approx(env,episodes=200,alpha=0.01,gamma=0.99,seed=seed,max_steps=500,feature_fn=feature_vector_linear)
    w_poly,poly_mse,poly_td=run_td0_linear_value_approx(env,episodes=200,alpha=0.01,gamma=0.99,seed=seed,max_steps=500,feature_fn=feature_vector_poly2)
    model,nn_loss,nn_td=run_td0_nn_value_approx(env,episodes=200,lr=0.001,gamma=0.99,seed=seed,max_steps=500,hidden=64,device="cuda")
    env.close()

    plot_learning_curves([lin_td,poly_td,nn_td],["Linear","Poly2","Neural"],"Lab10: Mean |TD Error|","Mean |TD Error|",window=10)
    plot_learning_curves([lin_mse,poly_mse,nn_loss],["Linear MSE","Poly2 MSE","NN Loss"],"Lab10: Loss Curves","Loss",window=10)

    stats={
        "linear_weights":w_lin,
        "poly_weights":w_poly,
        "linear_memory_bytes":approx_memory_bytes_linear(w_lin),
        "poly_memory_bytes":approx_memory_bytes_linear(w_poly),
        "nn_memory_bytes":approx_memory_bytes_network(model)
    }
    return stats


if __name__=="__main__":
    out=student_tasks_lab10(seed=0)
    print(out["linear_memory_bytes"],out["poly_memory_bytes"],out["nn_memory_bytes"])

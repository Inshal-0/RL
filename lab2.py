import numpy as np

# ===================== 1. GRID WORLD INITIALIZATION =====================

def init_gridworld(rows=3,cols=4,wall=(1,2),goal=(0,3),danger=(1,3)):
    """Create Grid World MDP components."""
    states=[(i,j) for i in range(rows) for j in range(cols)]
    if wall in states:
        states.remove(wall)
    actions=["UP","DOWN","LEFT","RIGHT"]
    mdp={
        "rows":rows,
        "cols":cols,
        "states":states,
        "actions":actions,
        "wall":wall,
        "goal":goal,
        "danger":danger
    }
    return mdp

# ===================== 2. REWARD FUNCTION =====================

def reward(state,goal,danger):
    """Reward function R(s)."""
    if state==goal:
        return 1.0
    elif state==danger:
        return -1.0
    else:
        return -0.04

# ===================== 3. DETERMINISTIC NEXT STATE =====================

def next_state(state,action,rows,cols,wall):
    """Deterministic transition before adding stochasticity."""
    i,j=state
    if action=="UP":
        i=max(i-1,0)
    elif action=="DOWN":
        i=min(i+1,rows-1)
    elif action=="LEFT":
        j=max(j-1,0)
    elif action=="RIGHT":
        j=min(j+1,cols-1)
    # If move hits wall → stay in same state
    if (i,j)==wall:
        return state
    return (i,j)

# ===================== 4. STOCHASTIC TRANSITION MODEL =====================

def transition_probabilities(state,action,mdp,p_intended=0.8,p_left=0.1,p_right=0.1):
    """
    Stochastic transition model with intended/slip probabilities.
    80-10-10 by default, but you can change to 70-15-15 etc.
    """
    rows=mdp["rows"]
    cols=mdp["cols"]
    wall=mdp["wall"]
    goal=mdp["goal"]
    danger=mdp["danger"]

    # Terminal states: agent stays there with prob 1
    if state in [goal,danger]:
        return {state:1.0}

    probs={}
    intended=next_state(state,action,rows,cols,wall)

    # Define left/right actions relative to current action
    if action=="UP":
        left,right="LEFT","RIGHT"
    elif action=="DOWN":
        left,right="RIGHT","LEFT"
    elif action=="LEFT":
        left,right="DOWN","UP"
    else:  # "RIGHT"
        left,right="UP","DOWN"

    slip_left=next_state(state,left,rows,cols,wall)
    slip_right=next_state(state,right,rows,cols,wall)

    probs[intended]=probs.get(intended,0)+p_intended
    probs[slip_left]=probs.get(slip_left,0)+p_left
    probs[slip_right]=probs.get(slip_right,0)+p_right

    return probs

# ===================== 5. PRINT TRANSITIONS FOR A STATE-ACTION =====================

def print_transitions(state,action,mdp,p_intended=0.8,p_left=0.1,p_right=0.1):
    """Utility to display transition probabilities and rewards for a given state,action."""
    goal=mdp["goal"]
    danger=mdp["danger"]
    trans=transition_probabilities(state,action,mdp,p_intended,p_left,p_right)
    print(f"From state {state}, action={action}:")
    for next_s,prob in trans.items():
        r=reward(next_s,goal,danger)
        print(f" -> {next_s} with P={prob:.2f}, Reward={r}")

# ===================== 6. GAMMA EXPERIMENT HELPER =====================

def set_gamma(gamma):
    """Simple helper if you want to play with gamma values."""
    return gamma  # mainly placeholder, gamma is used in value iteration etc.

# ===================== 7. VALUE ITERATION (ADVANCED TASK) =====================

def value_iteration(mdp,gamma=0.9,theta=1e-4,
                    p_intended=0.8,p_left=0.1,p_right=0.1):
    """
    Value Iteration to compute optimal state values and policy.
    mdp: dict from init_gridworld
    gamma: discount factor
    theta: convergence threshold
    """
    states=mdp["states"]
    actions=mdp["actions"]
    goal=mdp["goal"]
    danger=mdp["danger"]

    # Initialize values to zero
    V={s:0.0 for s in states}

    while True:
        delta=0.0
        for s in states:
            # Skip terminal states (values fixed by reward)
            if s==goal or s==danger:
                v_old=V[s]
                V[s]=reward(s,goal,danger)
                delta=max(delta,abs(v_old-V[s]))
                continue

            v_old=V[s]
            # Compute max over actions of sum_s' P(s'|s,a)[R(s')+gamma*V(s')]
            action_values=[]
            for a in actions:
                trans=transition_probabilities(s,a,mdp,p_intended,p_left,p_right)
                q_sa=0.0
                for s_next,p in trans.items():
                    r=reward(s_next,goal,danger)
                    q_sa+=p*(r+gamma*V[s_next])
                action_values.append(q_sa)
            V[s]=max(action_values)
            delta=max(delta,abs(v_old-V[s]))
        if delta<theta:
            break

    # Derive greedy policy π*(s)=argmax_a Q(s,a)
    policy={}
    for s in states:
        if s==goal or s==danger:
            policy[s]=None
            continue
        best_a=None
        best_q=-1e9
        for a in actions:
            trans=transition_probabilities(s,a,mdp,p_intended,p_left,p_right)
            q_sa=0.0
            for s_next,p in trans.items():
                r=reward(s_next,goal,danger)
                q_sa+=p*(r+gamma*V[s_next])
            if q_sa>best_q:
                best_q=q_sa
                best_a=a
        policy[s]=best_a

    return V,policy

# ===================== 8. PRETTY-PRINT VALUES AND POLICY =====================

def print_value_grid(V,mdp):
    rows=mdp["rows"]
    cols=mdp["cols"]
    wall=mdp["wall"]
    for i in range(rows):
        row_vals=[]
        for j in range(cols):
            s=(i,j)
            if s==wall:
                row_vals.append("  WALL  ")
            elif s in V:
                row_vals.append(f"{V[s]:7.3f}")
            else:
                row_vals.append("  N/A   ")
        print(" | ".join(row_vals))
    print()

def print_policy_grid(policy,mdp):
    rows=mdp["rows"]
    cols=mdp["cols"]
    wall=mdp["wall"]
    goal=mdp["goal"]
    danger=mdp["danger"]
    for i in range(rows):
        row_vals=[]
        for j in range(cols):
            s=(i,j)
            if s==wall:
                row_vals.append("WALL ")
            elif s==goal:
                row_vals.append("GOAL ")
            elif s==danger:
                row_vals.append("DANG ")
            else:
                a=policy.get(s,None)
                row_vals.append(f"{(a or 'None')[:4]:4}")
        print(" | ".join(row_vals))
    print()

# ===================== 9. QUICK DEMO CALLS (YOU CAN RUN THESE) =====================

if __name__=="__main__":
    mdp=init_gridworld()

    # Example 1: from (2,0), action="UP"
    print_transitions((2,0),"UP",mdp)

    # Example 2: from (0,2), action="RIGHT"
    print()
    print_transitions((0,2),"RIGHT",mdp)

    # Example slip change: 70-15-15
    print("\nWith 70-15-15 slip probabilities from (2,0), UP:")
    print_transitions((2,0),"UP",mdp,p_intended=0.7,p_left=0.15,p_right=0.15)

    # Value Iteration demo
    print("\nRunning Value Iteration with gamma=0.9...")
    V,policy=value_iteration(mdp,gamma=0.9)
    print("\nState Values:")
    print_value_grid(V,mdp)
    print("Optimal Policy:")
    print_policy_grid(policy,mdp)

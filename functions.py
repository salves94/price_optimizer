import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from matplotlib import animation, rc

from bokeh.io import show
from bokeh.palettes import PuBu4
from bokeh.plotting import figure
from bokeh.models import Label

## Environment simulator
def plus(x):
    return 0 if x < 0 else x

def minus(x):
    return 0 if x > 0 else -x

def shock(x):
    return np.sqrt(x)

# Demand at time step t for current price p_t and previous price p_t_1
def q_t(p_t, p_t_1, q_0, k, a, b):
    return plus(q_0 - k*p_t - a*shock(plus(p_t - p_t_1)) + b*shock(minus(p_t - p_t_1)))

# Profit at time step t
def profit_t(p_t, p_t_1, q_0, k, a, b, unit_cost):
    return q_t(p_t, p_t_1, q_0, k, a, b)*(p_t - unit_cost) 

# Total profit for price vector p over len(p) time steps
def profit_total(p, unit_cost, q_0, k, a, b):
    return profit_t(p[0], p[0], q_0, k, 0, 0, unit_cost) + sum(map(lambda t: profit_t(p[t], p[t-1], q_0, k, a, b, unit_cost), range(len(p))))

## Partial bindings for readability
def profit_t_response(p_t, p_t_1, q_0, k, increase_coefficient, decrease_coefficient, unit_cost):
    return profit_t(p_t, p_t_1, q_0, k, increase_coefficient, decrease_coefficient, unit_cost)

def profit_response(p, unit_cost, q_0, k, increase_coefficient, decrease_coefficient):
    return profit_total(p, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)

##################################################################################################################################

def environmentSimulator(max_price, price_step, q_0, k, unit_cost, increase_coefficient, decrease_coefficient, env_simulation_src):
    price_grid = np.arange(price_step, max_price, price_step)
    price_change_grid = np.arange(0.5, 2.0, 0.1)
    profit_map = np.zeros( (len(price_grid), len(price_change_grid)) )
    for i in range(len(price_grid)):
        for j in range(len(price_change_grid)):
            profit_map[i,j] = profit_t_response(price_grid[i], price_grid[i]*price_change_grid[j], q_0, k, increase_coefficient, decrease_coefficient, unit_cost)

    plt.figure(figsize=(16, 5))
    for i in range(len(price_change_grid)):
        if math.isclose(price_change_grid[i], 1.0):
            color = 'red'
        else:
            color = (0.6, 0.3, price_change_grid[i]/2.0)
        plt.plot(price_grid, profit_map[:, i], c=color)
    
    plt.xlabel("Price ($)")
    plt.ylabel("Profit")
    plt.legend(np.int_(np.round((1-price_change_grid)*100)), loc='lower right', title="Price change (%)", fancybox=False, framealpha=0.6)

    plt.savefig(env_simulation_src)
    return price_grid

def optimalConstantPrice(time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient):
    constant_profit = np.array([ profit_response(np.repeat(p, time_steps), unit_cost, q_0, k, increase_coefficient, decrease_coefficient) for p in price_grid ])
    p_idx = np.argmax(constant_profit)
    price_opt_const = price_grid[p_idx]

    return dict(price=price_opt_const, profit=constant_profit[p_idx])

def optimalSequenceOfPrices(optimal_constant_price, time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient, optimal_seq_price_src):
    prices = np.repeat(optimal_constant_price['price'], time_steps)
    for t in range(time_steps):
        price_t = findOptimalPriceT(prices, price_grid, t, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)
        prices[t] = price_t

    profit=profit_response(prices, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)

    plt.figure(figsize=(16, 5))
    plt.xlabel("Time step")
    plt.ylabel("Price ($)")
    plt.plot(range(len(prices)), prices, c='red')
    plt.savefig(optimal_seq_price_src)

    return dict(prices=prices, profit=profit)

def findOptimalPriceT(p_baseline, price_grid, t, unit_cost, q_0, k, increase_coefficient, decrease_coefficient):
  p_grid = np.tile(p_baseline, (len(price_grid), 1))
  p_grid[:, t] = price_grid
  profit_grid = np.array([ profit_response(p, unit_cost, q_0, k, increase_coefficient, decrease_coefficient) for p in p_grid ])
  return price_grid[ np.argmax(profit_grid) ]

def formatCurrency(value):
    return "${:,.2f}".format(value)

############################################################################################################################

def plot_return_trace(returns, returns_variation_src, smoothing_window=10, range_std=2):
    plt.figure(figsize=(16, 5))
    plt.xlabel("Episode")
    plt.ylabel("Return ($)")
    returns_df = pd.Series(returns)
    ma = returns_df.rolling(window=smoothing_window).mean()
    mstd = returns_df.rolling(window=smoothing_window).std()
    plt.plot(ma, c = 'blue', alpha = 1.00, linewidth = 1)
    plt.fill_between(mstd.index, ma-range_std*mstd, ma+range_std*mstd, color='blue', alpha=0.2)
    plt.savefig(returns_variation_src)

def plot_price_schedules(p_trace, sampling_ratio, last_highlights, T, price_schedules_src, fig_number=None):
    plt.figure(fig_number);
    plt.xlabel("Time step");
    plt.ylabel("Price ($)");
    plt.plot(range(T), np.array(p_trace[0:-1:sampling_ratio]).T, c = 'k', alpha = 0.05)
    plt.plot(range(T), np.array(p_trace[-(last_highlights+1):-1]).T, c = 'red', alpha = 0.5, linewidth=2)
    plt.savefig(price_schedules_src)

def bullet_graph(data, td_errors_src, labels=None, bar_label=None, axis_label=None, size=(5, 3), palette=None, bar_color="black", label_color="gray"):
    stack_data = np.stack(data[:,2])

    cum_stack_data = np.cumsum(stack_data, axis=1)
    h = np.max(cum_stack_data) / 20

    fig, axarr = plt.subplots(len(data), figsize=size, sharex=True)

    for idx, item in enumerate(data):

        if len(data) > 1:
            ax = axarr[idx]

        ax.set_aspect('equal')
        ax.set_yticklabels([item[0]])
        ax.set_yticks([1])
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        prev_limit = 0
        for idx2, lim in enumerate(cum_stack_data[idx]):
            ax.barh([1], lim - prev_limit, left=prev_limit, height=h, color=palette[idx2])
            prev_limit = lim
        rects = ax.patches
        ax.barh([1], item[1], height=(h / 3), color=bar_color)

    if labels is not None:
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                -height * .4,
                label,
                ha='center',
                va='bottom',
                color=label_color)
            
    if bar_label is not None:
        rect = rects[0]
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width(),
            -height * .1,
            bar_label,
            ha='center',
            va='bottom',
            color='white')
    if axis_label:
        ax.set_xlabel(axis_label)
    fig.subplots_adjust(hspace=0)
    fig.savefig(td_errors_src)
    
########################################################   DQN   ###########################################################

from collections import namedtuple
from itertools import count
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# A cyclic buffer of bounded size that holds the transitions observed recently
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PolicyNetworkDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
      super(PolicyNetworkDQN, self).__init__()
      layers = [
              nn.Linear(state_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, action_size)
      ]
      self.model = nn.Sequential(*layers)

    def forward(self, x):
      q_values = self.model(x)
      return q_values  

class AnnealedEpsGreedyPolicy(object):
  def __init__(self, eps_start = 0.9, eps_end = 0.05, eps_decay = 400):
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.steps_done = 0

  def select_action(self, q_values):
    sample = random.random()
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
    self.steps_done += 1
    if sample > eps_threshold:
        return np.argmax(q_values)
    else:
        return random.randrange(len(q_values))

def update_model(memory, policy_net, target_net, device, optimizer, gamma, batch_size):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = reward_batch[:, 0] + (gamma * next_state_values)  

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def env_intial_state(T):
    return np.repeat(0, 2*T)

def env_step(t, state, action, price_grid, T, q_0, k, increase_coefficient, decrease_coefficient, unit_cost):
    next_state = np.repeat(0, len(state))
    next_state[0] = price_grid[action]
    next_state[1:T] = state[0:T-1]
    next_state[T+t] = 1
    reward = profit_t_response(next_state[0], next_state[1], q_0, k, increase_coefficient, decrease_coefficient, unit_cost)
    return next_state, reward

def to_tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32))

def to_tensor_long(x, device):
    return torch.tensor([[x]], device=device, dtype=torch.long)

def deepQN(price_grid, T, device, q_0, k, increase_coefficient, decrease_coefficient, unit_cost, gamma, target_update, batch_size, learning_rate, num_episodes, returns_variation_src, price_schedules_src):
    policy_net = PolicyNetworkDQN(2*T, len(price_grid)).to(device)
    target_net = PolicyNetworkDQN(2*T, len(price_grid)).to(device)
    optimizer = optim.AdamW(policy_net.parameters(), lr = learning_rate)
    policy = AnnealedEpsGreedyPolicy()
    memory = ReplayMemory(10000)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    return_trace = []
    p_trace = [] # price schedules used in each episode
    for i_episode in range(num_episodes):
        state = env_intial_state(T)
        reward_trace = []
        p = []
        for t in range(T):
            # Select and perform an action
            with torch.no_grad():
                q_values = policy_net(to_tensor(state))
            action = policy.select_action(q_values.detach().numpy())

            next_state, reward = env_step(t, state, action, price_grid, T, q_0, k, increase_coefficient, decrease_coefficient, unit_cost)

            # Store the transition in memory
            memory.push(to_tensor(state), 
                        to_tensor_long(action, device), 
                        to_tensor(next_state) if t != T - 1 else None, 
                        to_tensor([reward]))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            update_model(memory, policy_net, target_net, device, optimizer, gamma, batch_size)

            reward_trace.append(reward)
            p.append(price_grid[action])

        return_trace.append(sum(reward_trace))
        p_trace.append(p)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

            #clear_output(wait = True)
            print(f'Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)')

    plot_return_trace(return_trace, returns_variation_src)

    fig = plt.figure(figsize=(16, 5))
    plot_price_schedules(p_trace, 5, 1, T, price_schedules_src, fig.number)
    
    return dict(profit=sorted(profit_response(s, unit_cost, q_0, k, increase_coefficient, decrease_coefficient) for s in p_trace)[-10:], memory=memory, policy_net=policy_net, target_net=target_net, policy=policy)

################################################## Policy visualization, tuning, and debugging #####################################################

# TD error Debugging Q-values computations
def TDError(gamma, device, memory, policy_net, target_net, td_errors_src):

    transitions = memory.sample(10)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(len(transitions), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * gamma) + reward_batch[:, 0]

    q_trace = []
    for t in range(len(transitions)):
        print(f"Q_(s,a)[ {expected_state_action_values[t]} ] = r [ {reward_batch[t].item()} ] + g*Q_(s+1)[ {next_state_values[t]} ]  <> Q_(s,a)[ {state_action_values[t].item()} ]")
        q_trace.append([f"Sample {t}", state_action_values[t].item(), [reward_batch[t].item(), next_state_values[t]]])

    palette = sns.light_palette("crimson", 3, reverse=False)
    bullet_graph(np.array(q_trace), td_errors_src,
                labels=["r", "max_a' Q(s', a')"], bar_label="Q(s, a)", size=(20, 10),
                axis_label="Q-value ($)", label_color="black",
                bar_color="#252525", palette=palette)

    return q_trace

def correlation(T, gamma, policy_net, policy, price_grid, q_0, k, increase_coefficient, decrease_coefficient, unit_cost, correlation_src):
    num_episodes = 100
    return_trace = []
    q_values_rewards_trace = np.zeros((num_episodes, T, 2, ))
    for i_episode in range(num_episodes):        # modified copy of the simulation loop
        state = env_intial_state(T)
        for t in range(T):
            # Select and perform an action
            with torch.no_grad():
                q_values = policy_net(to_tensor(state)).detach().numpy()
            action = policy.select_action(q_values)

            next_state, reward = env_step(t, state, action, price_grid, T, q_0, k, increase_coefficient, decrease_coefficient, unit_cost)

            # Move to the next state
            state = next_state

            q_values_rewards_trace[i_episode][t][0] = q_values[action]
            for tau in range(t):
                q_values_rewards_trace[i_episode][tau][1] += reward * (gamma ** (t - tau)) 


    # Visualizing the distribution of Q-value vs actual returns 
    values = np.reshape(q_values_rewards_trace, (num_episodes * T, 2, ))

    df = pd.DataFrame(data=values, columns=['Q-value', 'Return'])
    g = sns.jointplot(x="Q-value", y="Return", data=df, kind="kde", color="crimson", height=10, aspect=1.0)
    g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+", alpha=0.1)
    g.ax_joint.collections[0].set_alpha(0)

    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ':k')   

    g.savefig(correlation_src)
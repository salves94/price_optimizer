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

def environmentSimulator(max_price, price_step, q_0, k, unit_cost, increase_coefficient, decrease_coefficient):
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

    plt.savefig('static/images/plot.png')
    return price_grid

def optimalConstantPrice(time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient):
    constant_profit = np.array([ profit_response(np.repeat(p, time_steps), unit_cost, q_0, k, increase_coefficient, decrease_coefficient) for p in price_grid ])
    p_idx = np.argmax(constant_profit)
    price_opt_const = price_grid[p_idx]

    return dict(price=price_opt_const, profit=constant_profit[p_idx])

def optimalSequenceOfPrices(optimal_constant_price, time_steps, price_grid, unit_cost, q_0, k, increase_coefficient, decrease_coefficient):
    prices = np.repeat(optimal_constant_price['price'], time_steps)
    for t in range(time_steps):
        price_t = findOptimalPriceT(prices, price_grid, t, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)
        prices[t] = price_t

    profit=profit_response(prices, unit_cost, q_0, k, increase_coefficient, decrease_coefficient)

    plt.figure(figsize=(16, 5))
    plt.xlabel("Time step")
    plt.ylabel("Price ($)")
    plt.plot(range(len(prices)), prices, c='red')
    plt.savefig('static/images/plot2.png')

    return dict(prices=prices, profit=profit)

def findOptimalPriceT(p_baseline, price_grid, t, unit_cost, q_0, k, increase_coefficient, decrease_coefficient):
  p_grid = np.tile(p_baseline, (len(price_grid), 1))
  p_grid[:, t] = price_grid
  profit_grid = np.array([ profit_response(p, unit_cost, q_0, k, increase_coefficient, decrease_coefficient) for p in p_grid ])
  return price_grid[ np.argmax(profit_grid) ]

def formatCurrency(value):
    return "${:,.2f}".format(value)

############################################################################################################################

def plot_return_trace(returns, smoothing_window=10, range_std=2):
    plt.figure(figsize=(16, 5))
    plt.xlabel("Episode")
    plt.ylabel("Return ($)")
    returns_df = pd.Series(returns)
    ma = returns_df.rolling(window=smoothing_window).mean()
    mstd = returns_df.rolling(window=smoothing_window).std()
    plt.plot(ma, c = 'blue', alpha = 1.00, linewidth = 1)
    plt.fill_between(mstd.index, ma-range_std*mstd, ma+range_std*mstd, color='blue', alpha=0.2)
    plt.savefig('static/images/plot3.png')

def plot_price_schedules(p_trace, sampling_ratio, last_highlights, T, fig_number=None):
    plt.figure(fig_number);
    plt.xlabel("Time step");
    plt.ylabel("Price ($)");
    plt.plot(range(T), np.array(p_trace[0:-1:sampling_ratio]).T, c = 'k', alpha = 0.05)
    plt.plot(range(T), np.array(p_trace[-(last_highlights+1):-1]).T, c = 'red', alpha = 0.5, linewidth=2)
    plt.savefig('static/images/plot4.png')
    
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

def deepQN(price_grid, T, device, q_0, k, increase_coefficient, decrease_coefficient, unit_cost, gamma, target_update, batch_size, learning_rate, num_episodes):
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

    plot_return_trace(return_trace)

    fig = plt.figure(figsize=(16, 5))
    plot_price_schedules(p_trace, 5, 1, T, fig.number)
    
    return sorted(profit_response(s, unit_cost, q_0, k, increase_coefficient, decrease_coefficient) for s in p_trace)[-10:]
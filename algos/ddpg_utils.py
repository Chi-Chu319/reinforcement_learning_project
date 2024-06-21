
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
import numpy as np
import torch

# from torch.distributions import Categorical
from torch.distributions import Normal, Independent

import pickle, os, random, torch

from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt

Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])

class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, supports, num_atoms=51, v_min=-10, v_max=10):
        """
        Distributional critic for estimating the distribution of Q-values for state-action pairs.

        Parameters:
        - `state_dim`: Dimensionality of the state space.
        - `action_dim`: Dimensionality of the action space.
        - `supports`: The supports for the distribution.
        - `num_atoms`: The number of atoms in the distribution.
        - `v_min`: Minimum value for the distributional critic.
        - `v_max`: Maximum value for the distributional critic.
        """
        super(DistributionalCritic, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, num_atoms))
        self.supports = supports
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, num_atoms]

    def get_probs(self, state, action):
        return torch.softmax(self.forward(state, action), dim=1)

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """
        Actor network for the actor-critic agent. Responsible for producing actions based on observed states.

        Parameters:
        - `state_dim`: Dimensionality of the state space.
        - `action_dim`: Dimensionality of the action space.
        - `max_action`: Maximum absolute value for each action component.
        """
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Critic network for the actor-critic agent. Estimates the value function for state-action pairs.

        Parameters:
        - `state_dim`: Dimensionality of the state space.
        - `action_dim`: Dimensionality of the action space.
        """
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]

class ReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, max_size=int(1e6)):
        """
        Replay buffer to store and sample experiences for training the agent. Supports additional information in the form of extra dictionaries.

        Parameters:
        - `state_shape`: Tuple representing the shape of the state space.
        - `action_dim`: Dimensionality of the action space.
        - `max_size`: Maximum size of the replay buffer (default: 1e6).
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        dtype = torch.uint8 if len(state_shape) == 3 else torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros((max_size, *state_shape), dtype=dtype)
        self.reward = torch.zeros((max_size, 1), dtype=dtype)
        self.not_done = torch.zeros((max_size, 1), dtype=dtype)
        self.extra = {}
    
    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, extra:dict=None):
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[self.ptr] = self._to_tensor(reward)
        self.not_done[self.ptr] = self._to_tensor(1. - done)

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = torch.zeros((self.max_size, *value.shape), dtype=torch.float32)
                self.extra[key][self.ptr] = self._to_tensor(value)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device='cpu'):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.extra:
            extra = {key: value[ind].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[ind].to(device),
            action = self.action[ind].to(device), 
            next_state = self.next_state[ind].to(device), 
            reward = self.reward[ind].to(device), 
            not_done = self.not_done[ind].to(device), 
            extra = extra
        )
        return batch
    
    def get_all(self, device='cpu'):
        if self.extra:
            extra = {key: value[:self.size].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[:self.size].to(device),
            action = self.action[:self.size].to(device), 
            next_state = self.next_state[:self.size].to(device), 
            reward = self.reward[:self.size].to(device), 
            not_done = self.not_done[:self.size].to(device), 
            extra = extra
        )
        return batch
    
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_shape: tuple, action_dim: int, max_size=int(1e6), alpha=0.6, beta=0.4):
        super(PrioritizedReplayBuffer, self).__init__(state_shape, action_dim, max_size) 
        """
        Prioritized replay buffer that assigns priorities to experiences based on their temporal difference errors. Inherits from `ReplayBuffer`.

        Parameters:
        - `state_shape`: Tuple representing the shape of the state space.
        - `action_dim`: Dimensionality of the action space.
        - `max_size`: Maximum size of the replay buffer.
        - `alpha`: Priority exponent controlling the amount of prioritization.
        - `beta`: Importance sampling exponent for adjusting the bias of the updates.
        """
        self.priorities = np.empty(max_size, dtype=np.float32)

        self.alpha = alpha
        self.beta = beta

    def add(self, state, action, next_state, reward, done):
        priority = 1.0 if self.size == 0 else np.max(self.priorities[:self.size])

        if self.size == self.max_size:
            if priority > np.min(self.priorities):
                min_priority_idx = np.argmin(self.priorities)
                self.priorities[min_priority_idx] = priority
                self.state[min_priority_idx] = self._to_tensor(state, dtype=self.state.dtype)
                self.action[min_priority_idx] = self._to_tensor(action)
                self.next_state[min_priority_idx] = self._to_tensor(next_state, dtype=self.state.dtype)
                self.reward[min_priority_idx] = self._to_tensor(reward)
                self.not_done[min_priority_idx] = self._to_tensor(1. - done)
            else:
                pass # Do not add low priority experiences
        else: 
            self.priorities[self.ptr] = priority
            self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
            self.action[self.ptr] = self._to_tensor(action)
            self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
            self.reward[self.ptr] = self._to_tensor(reward)
            self.not_done[self.ptr] = self._to_tensor(1. - done)

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            
            
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def sample(self, batch_size, device="cpu"):
        priorities = self.priorities[:self.size]
        prob = priorities ** self.alpha / np.sum(priorities ** self.alpha)
        
        indices = np.random.choice(np.arange(self.size), size=batch_size, p=prob)

        weights = (self.size * prob[indices]) ** (-self.beta)
        weights /= weights.max()

        if self.extra:
            extra = {key: value[indices].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[indices].to(device),
            action = self.action[indices].to(device),
            next_state = self.next_state[indices].to(device),
            reward = self.reward[indices].to(device),
            not_done = self.not_done[indices].to(device),
            extra = extra
        )
        return batch, indices, torch.from_numpy(weights).to(device)
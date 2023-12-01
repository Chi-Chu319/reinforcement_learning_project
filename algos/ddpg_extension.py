import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .ddpg_agent import DDPGAgent
from .ddpg_utils import PrioritizedReplayBuffer, Policy, Critic
import utils.common_utils as cu
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path

class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        super(DDPGExtension, self).__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q2 = Critic(self.observation_space_dim, self.action_space_dim).to(self.device)
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=float(self.lr))

        self.policy_noise = 0.2*self.max_action
        self.noise_clip = 0.5*self.max_action
        self.policy_delay = 2

        self.buffer = PrioritizedReplayBuffer(state_shape=(self.observation_space_dim,), 
                                              action_dim=self.action_dim, max_size=int(float(self.buffer_size)))

    def _update(self):
        batch = self.buffer.sample(self.batch_size, device=self.device)
        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done
        
        noise = torch.normal(0, self.policy_noise, size=action.size()).to(self.device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_action = self.pi_target(next_state) + noise
        target_Q1 = self.q_target(next_state, next_action)
        target_Q2 = self.q2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + self.gamma * not_done * target_Q

        current_Q1 = self.q(state, action)
        current_Q2 = self.q2(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.q_optim.zero_grad()
        self.q2_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()
        self.q2_optim.step()

        if self.buffer_ptr % self.policy_delay == 0:
            actor_loss = -self.q(state, self.pi(state)).mean()

            self.pi_optim.zero_grad()
            actor_loss.backward()
            self.pi_optim.step()

            cu.soft_update_params(self.q, self.q_target, self.tau)
            cu.soft_update_params(self.q, self.q2_target, self.tau)
            cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}
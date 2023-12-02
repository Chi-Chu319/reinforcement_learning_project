import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from algos.ddpg_agent import DDPGAgent
from algos.ddpg_utils import Policy, ReplayBuffer
import copy, time
import utils.common_utils as cu

class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, supports, num_atoms=51, v_min=-10, v_max=10):
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

class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        super(DDPGExtension, self).__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.v_min = -self.cfg.env_config["n_no_sanding"]
        self.v_max = self.cfg.env_config["n_sanding"]
        self.num_atoms = self.cfg.num_atoms
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.supports = torch.arange(self.v_min, self.v_max + self.delta_z, self.delta_z).to(self.device)

        # Override the Critic and Policy networks
        self.q = DistributionalCritic(
            self.observation_space_dim,
            self.action_space_dim,
            supports = self.supports,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)

        self.q_target = copy.deepcopy(self.q)

        self.pi = Policy(self.observation_space_dim, self.action_space_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))

        self.critic_loss = nn.BCELoss(reduction='none')

        self.buffer = ReplayBuffer(
            state_shape=(self.observation_space_dim,),
            action_dim=self.action_dim,
            max_size=int(float(self.buffer_size))
        )

    def _get_critic_loss(self, state, action, next_state, reward, not_done, current_Q_dist, next_Q_dist):
        target_z = reward + not_done * self.gamma * self.supports
        target_z = target_z.clamp(min=self.v_min, max=self.v_max)

        b = (target_z - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1

        proj_dist = torch.zeros_like(next_Q_dist)
        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).unsqueeze(1).expand(self.batch_size, self.num_atoms).long().to(self.device)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_Q_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_Q_dist * (b - l.float())).view(-1))

        log_p = torch.log(current_Q_dist)

        loss = -(log_p * proj_dist).sum(-1).mean()

        return loss

    def _update(self):
        batch = self.buffer.sample(self.batch_size, device=self.device)

        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        # critic loss
        with torch.no_grad():
            next_Q_dist = self.q_target.get_probs(next_state, self.pi_target(next_state))

        current_Q_dist = self.q.get_probs(state, action)

        critic_loss = self._get_critic_loss(state, action, next_state, reward, not_done, current_Q_dist, next_Q_dist)

        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # actor loss
        sampled_actions = self.pi(state)
        sampled_Q_values = self.q(state, sampled_actions)
        actor_loss = -self.q.distr_to_q(sampled_Q_values)
        actor_loss = actor_loss.mean()

        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}
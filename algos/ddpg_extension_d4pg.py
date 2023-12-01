import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from algos.ddpg_agent import DDPGAgent
from algos.ddpg_utils import Policy, ReplayBuffer

class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10):
        super(DistributionalCritic, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.fc1 = nn.Linear(state_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_atoms)

        delta = (v_max - v_min) / (num_atoms - 1)
        self.supports = torch.arange(v_min, v_max + delta, delta).to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = F.softmax(x, dim=1)
        return x, probs

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
        # Override the Critic and Policy networks
        self.q = DistributionalCritic(self.observation_space_dim, self.action_space_dim,
                                      num_atoms=self.num_atoms,
                                      v_min=self.v_min, v_max=self.v_max).to(self.device)
        self.q_target = copy.deepcopy(self.q)

        self.pi = Policy(self.observation_space_dim, self.action_space_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))

        self.critic_loss = nn.BCELoss(reduction='none')

        self.buffer = ReplayBuffer(state_shape=(self.observation_space_dim,),
                                  action_dim=self.action_dim, max_size=int(float(self.buffer_size)))

    """
    def _l2_project(self, next_distr_v, rewards_v, dones_mask_t, gamma, delta_z, n_atoms, v_min, v_max):
        next_distr = next_distr_v.cpu().detach().numpy()
        rewards = rewards_v.cpu().detach().numpy()
        dones_mask = dones_mask_t.cpu().detach().numpy().astype(bool).flatten()
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
        for atom in range(n_atoms):
            tz_j = np.minimum(v_max, np.maximum(v_min, rewards + (v_min + atom * delta_z) * gamma))
            b_j = (tz_j - v_min) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_mask = eq_mask.flatten()
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            ne_mask = ne_mask.flatten()

            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones_mask]))
            b_j = (tz_j - v_min) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_mask = eq_mask.flatten()
            eq_dones = dones_mask.copy()
            eq_dones[dones_mask] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_mask = ne_mask.flatten()

            ne_dones = dones_mask.copy()
            ne_dones[dones_mask] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr).to(self.device)
    """

    def _l2_project(self, next_distr_v, rewards_v, dones_mask_t, gamma, delta_z, n_atoms, v_min, v_max):
      next_distr = next_distr_v.detach()
      rewards = rewards_v.detach()
      dones_mask = dones_mask_t.detach().bool().flatten()
      batch_size = len(rewards)
      proj_distr = torch.zeros((batch_size, n_atoms), dtype=torch.float32, device=self.device)

      for atom in range(n_atoms):
          tz_j = torch.clamp(rewards + (v_min + atom * delta_z) * gamma, min=v_min, max=v_max)
          b_j = (tz_j - v_min) / delta_z
          l = torch.floor(b_j).to(torch.int64)
          u = torch.ceil(b_j).to(torch.int64)
          eq_mask = u == l
          eq_mask = eq_mask.flatten()
          proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
          ne_mask = u != l
          ne_mask = ne_mask.flatten()

          proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
          proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
      if dones_mask.any():
          proj_distr[dones_mask] = 0.0
          tz_j = torch.clamp(rewards[dones_mask], min=v_min, max=v_max)

          b_j = (tz_j - v_min) / delta_z
          l = torch.floor(b_j).to(torch.int64)
          u = torch.ceil(b_j).to(torch.int64)
          eq_mask = u == l
          eq_mask = eq_mask.flatten()
          eq_dones = dones_mask.clone()
          eq_dones[dones_mask] = eq_mask
          if eq_dones.any():
              proj_distr[eq_dones, l[eq_mask]] = 1.0
          ne_mask = u != l
          ne_mask = ne_mask.flatten()

          ne_dones = dones_mask.clone()
          ne_dones[dones_mask] = ne_mask
          if ne_dones.any():
              proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
              proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

      return proj_distr

    def _update(self):
        batch = self.buffer.sample(self.batch_size, device=self.device)

        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        target_Q, target_Q_dist = self.q_target(next_state, self.pi_target(next_state))
        target_Q_dist = target_Q_dist.to(self.device)
        
        target_z_projected = self._l2_project(next_distr_v=target_Q_dist,
                                              rewards_v=reward,
                                              dones_mask_t=not_done,
                                              gamma=self.gamma,
                                              delta_z=self.delta_z,
                                              n_atoms=self.num_atoms,
                                              v_min=self.v_min,
                                              v_max=self.v_max)
        
        current_Q, current_Q_dist = self.q(state, action)
        current_Q_dist = current_Q_dist.to(self.device)
        """
        current_z_projected = self._l2_project(next_distr_v=current_Q_dist,
                                              rewards_v=reward,
                                              dones_mask_t=not_done,
                                              gamma=self.gamma,
                                              delta_z=self.delta_z,
                                              n_atoms=self.num_atoms,
                                              v_min=self.v_min,
                                              v_max=self.v_max)
        """
        #prob_dist_v = -F.log_softmax(current_Q, dim=1) * target_z_projected
        #critic_loss = prob_dist_v.sum(dim=1).mean()
        #critic_loss = self.critic_loss(current_z_projected, target_z_projected.detach()).sum(axis=1).mean(axis=0) #CHECK THIS
        #critic_loss.requires_grad = True
        prob_dist_v = -F.log_softmax(current_Q, dim=1) * target_z_projected.detach()
        critic_loss = prob_dist_v.sum(dim=1).mean()

        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        sampled_actions = self.pi(state)
        sampled_Q_values, sampled_Q_distribution = self.q(state, sampled_actions)
        #actor_loss = -torch.sum(sampled_Q_distribution * torch.tensor(self.q.z_atoms, device=self.device, dtype=torch.float), dim=-1)
        #actor_loss = actor_loss.mean()
        actor_loss = -self.q.distr_to_q(sampled_Q_distribution)
        actor_loss = actor_loss.mean()

        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}
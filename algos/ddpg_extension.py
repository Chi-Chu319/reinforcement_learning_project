import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from algos.ddpg_agent import DDPGAgent
from algos.ddpg_utils import Policy, DistributionalCritic, PrioritizedReplayBuffer
import copy, time
import utils.common_utils as cu

class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        """
        An extension of the DDPGAgent that uses a Distributional Critic and Prioritized Replay Buffer.

        Parameters:
            - config: Configuration object containing hyperparameters and environment settings.

        Attributes:
            - device: The device (cuda or cpu) on which the agent is running.
            - name: A string identifier for the agent.
            - v_min: Minimum value for the distributional critic's support.
            - v_max: Maximum value for the distributional critic's support.
            - num_atoms: Number of atoms in the distributional critic's output.
            - delta_z: The spacing between atoms in the distributional critic's output.
            - supports: Tensor representing the support for the distributional critic.
            - q: Distributional Critic network.
            - q_target: Target Distributional Critic network.
            - pi: Policy network.
            - pi_target: Target Policy network.
            - pi_optim: Optimizer for the Policy network.
            - q_optim: Optimizer for the Distributional Critic network.
            - buffer: Prioritized Replay Buffer for storing experiences.

        Methods:
            - _get_critic_loss: Compute the critic loss for the distributional critic.
            - _update: Perform a single update step for both the actor and the critic networks.
        """
        super(DDPGExtension, self).__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg_extension'

        self.v_min = -self.cfg.env_config["n_no_sanding"]
        self.v_max = self.cfg.env_config["n_sanding"]
        self.num_atoms = 51
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

        self.buffer = PrioritizedReplayBuffer(
            state_shape=(self.observation_space_dim,),
            action_dim=self.action_dim,
            max_size=int(float(self.buffer_size))
        )

    def _get_critic_loss(self, state, action, next_state, reward, not_done, weights, current_Q_dist, next_Q_dist):
        """
        Compute the critic loss for the distributional critic.
        Parameters:
        - state: Current state tensor.
        - action: Action tensor.
        - next_state: Next state tensor.
        - reward: Reward tensor.
        - not_done: Not done tensor (1 for not done, 0 for done).
        - weights: Importance sampling weights for prioritized replay.
        - current_Q_dist: Distribution of Q-values for the current state-action pair.
        - next_Q_dist: Distribution of Q-values for the next state-action pair.

        Returns:
        - loss: Critic loss tensor.
        """
        
        # Calculate the target distribution for the distributional critic
        target_z = reward + not_done * self.gamma * self.supports
        target_z = target_z.clamp(min=self.v_min, max=self.v_max)
        
        # Map the target values to the corresponding atoms in the distribution
        b = (target_z - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1
        
        # Project the next state distribution onto the target distribution
        proj_dist = torch.zeros_like(next_Q_dist)
        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).unsqueeze(1).expand(self.batch_size, self.num_atoms).long().to(self.device)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_Q_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_Q_dist * (b - l.float())).view(-1))
        
        # Calculate the log probabilities of the current state distribution
        log_p = torch.log(current_Q_dist)

        # Compute the critic loss using the cross-entropy loss with importance sampling weights
        loss = -((log_p * proj_dist) * weights.unsqueeze(1).detach()).sum(-1).mean()

        return loss

    def _update(self):
        """
        Perform a single update step for both the actor and the critic networks.
        """
        batch, indices, weights = self.buffer.sample(self.batch_size, device=self.device)

        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        # Critic loss
        with torch.no_grad():
            next_Q_dist = self.q_target.get_probs(next_state, self.pi_target(next_state))

        current_Q_dist = self.q.get_probs(state, action)

        critic_loss = self._get_critic_loss(state, action, next_state, reward, not_done, weights, current_Q_dist, next_Q_dist)

        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # Actor loss
        sampled_actions = self.pi(state)
        sampled_Q_values = self.q(state, sampled_actions)
        actor_loss = -self.q.distr_to_q(sampled_Q_values)
        actor_loss = actor_loss.mean()

        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # Update priorities for sampled batch
        # https://danieltakeshi.github.io/2019/07/14/per/
        '''
        we use |δi| as the magnitude of the TD error.
        Negative versus positive TD errors are combined into one case here,
        but in principle we could consider them as separate cases and add a bonus to whichever one we feel is more important to address.
        '''
        with torch.no_grad():
            target_Q = reward + self.gamma * not_done * self.q_target.distr_to_q(next_Q_dist)
            deltas = (target_Q - self.q.distr_to_q(current_Q_dist)).squeeze().abs()

            self.buffer.update_priorities(indices, deltas + 1e-6)

        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}
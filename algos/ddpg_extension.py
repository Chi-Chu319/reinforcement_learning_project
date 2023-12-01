import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .ddpg_agent import DDPGAgent
from .ddpg_utils import PrioritizedReplayBuffer,OrnsteinUhlenbeckProcess
import utils.common_utils as cu
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path


class DDPGExtension(DDPGAgent):
    def __init__(self, config=None):
        super(DDPGExtension, self).__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.buffer = PrioritizedReplayBuffer(state_shape=(self.observation_space_dim,), 
                                              action_dim=self.action_dim, max_size=int(float(self.buffer_size)))
        self.noise_process = OrnsteinUhlenbeckProcess(size=self.action_dim, sigma=0.2)
        
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)


        
        # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
        # Hint: Make sure the returned action's shape is correct.
        if not evaluation and self.buffer_ptr < self.random_transition: # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim)
        else:
            action = self.pi(x)
            if not evaluation:
                expl_noise = torch.tensor(self.noise_process.sample(), dtype=torch.float32).to(self.device)
                action += expl_noise

        #action = torch.clip(action, -self.max_action, self.max_action)

        return action, {} # just return a positional value

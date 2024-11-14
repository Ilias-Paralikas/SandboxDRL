import torch 
import torch.nn as nn
import numpy as np
class Model(nn.Module):
    def __init__(self,env):
        super(Model, self).__init__()
        self.in_features = int(np.prod(env.observation_space.shape))
        self.action_space = env.action_space.n
        self.net = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space)
        )
    def forward(self, x):
        return self.net(x)
        
        
    def get_action_space(self):
        return self.action_space
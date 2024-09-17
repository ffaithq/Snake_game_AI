
import torch
import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, n_actions)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(F.relu(self.layer1(x)))
        x = self.drop(F.relu(self.layer2(x)))
        return self.layer3(x)
    

    def save(self,path):
        torch.save(self.state_dict(),f'{path}')
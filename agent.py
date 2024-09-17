import random
import torch
import torch.optim as optim
import torch.nn as nn

class Agent:
    def __init__(self,arch,n_state,n_action,optimizer,lr):
        self.target_net = arch(n_state,n_action)
        self.policy_net = arch(n_state,n_action)

        self.init_model()
        self.optimizer = self.get_optimizer(optimizer,lr) 
    
    def get_optimizer(self,optimizer,lr):

        if optimizer == 'AdamW':
            return optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        if optimizer == 'Adam':
            return optim.Adam(self.policy_net.parameters(), lr=lr)

    def init_model(self):
        def init_weights(m):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param.data)
                else:
                    nn.init.constant_(param.data, 0)
        
        self.policy_net.apply(init_weights)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self,batch,gamma,criterion):
        batch_size = len(batch.state)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def update_model(self,aplha):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*aplha + target_net_state_dict[key]*(1-aplha)
        self.target_net.load_state_dict(target_net_state_dict)

    def action(self,epsilon,state,train=True):
        if train:
            if random.uniform(0,1) < epsilon:
                return torch.tensor([[random.randint(0,3)]])
            else:
                return self.policy_net(state).argmax(1).view(1, 1)
        return self.target_net(state).argmax(1).view(1, 1)

    def eval(self,path):
        self.target_net.load_state_dict(torch.load(path,weights_only=True))
        self.target_net.eval()

    def save_model(self,path_target,path_policy):
        self.target_net.save(f"{path_target}")
        self.policy_net.save(f"{path_policy}")
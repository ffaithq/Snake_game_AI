from memory import Transition
from tqdm import tqdm
import torch
from agent import Agent
from snake import Snake
import matplotlib.pyplot as plt
from history import Tracker

class DQLearning:
    def __init__(self,env,agent,memory,hyperparametrs,tracker):
        self.env = env
        self.agent = agent
        self.memory = memory
        self.hyperparametrs = hyperparametrs
        self.history = tracker
        #self.wandb = wandb.init()
    

    def get_samples(self):
        if len(self.memory) < self.hyperparametrs['batch_size']:
            return None
        transition = self.memory.sample(self.hyperparametrs['batch_size'])
        batch = Transition(*zip(*transition))
        if not tuple(map(lambda s: s is not None,batch.next_state)) and not [s for s in batch.next_state if s is not None]:
            return None

        return batch

    def train(self,episods,criterion,path_target,path_policy):
        for i in tqdm(range(episods)):
            self.env.reset()
            done = False
            cumulative_reward = 0
            time_step = 0 
            loss = 0
            while not done:

                state = self.env.get_state()
                state = torch.tensor(state,dtype=torch.float,device=self.agent.device).unsqueeze(0)
                action = self.agent.action(self.hyperparametrs['epsilon'],state)


                done,reward,info = self.env.step(action=action.item())
                cumulative_reward += reward

                reward = torch.tensor([reward],dtype=torch.float,device=self.agent.device)
                
                if done:
                    new_state = None
                else:
                    new_state = torch.tensor(self.env.get_state(),dtype=torch.float,device=self.agent.device).unsqueeze(0)

                self.memory.push(state,action,new_state,reward)

                state = new_state
                batch = self.get_samples()
                if batch:
                    loss += self.agent.optimize_model(batch,self.hyperparametrs['gamma'],criterion)
                    self.agent.update_model(self.hyperparametrs['alpha'])

                time_step += 1

 
            self.history.push(self.env.get_score(),'Train score')
            self.history.push(time_step,'Train time step')
            self.history.push(cumulative_reward/time_step,'Train average reward')
            self.history.push(cumulative_reward,'Train cumulative reward')
            

            self.agent.save_model(path_target,path_policy)

            if i & 50 == 0:
                self.history.plot('Train score','Train time step','Train average reward','Train cumulative reward')
                self.history.save()

    def eval(self,num_episod,path):
        self.agent.eval(path)
        self.env.render()
        for i in tqdm(range(num_episod)):
            self.env.reset()
            done = False
            cumulative_reward = 0
            time_step = 0 
            while not done:

                state = self.env.get_state()
                state = torch.tensor(state,dtype=torch.float,device=self.agent.device).unsqueeze(0)
                action = self.agent.action(self.hyperparametrs['epsilon'],state,False)
                done,reward,reason = self.env.step(action=action.item())
                
                if done:
                    new_state = None
                else:
                    new_state = torch.tensor(self.env.get_state(),dtype=torch.float).unsqueeze(0)

                state = new_state

                cumulative_reward += reward
                time_step += 1

                
            self.history.push(self.env.get_score(),'Eval score')
            self.history.push(time_step,'Eval time step')
            self.history.push(cumulative_reward/time_step,'Eval average reward')
            self.history.push(cumulative_reward,'Eval cumulative reward')
            self.history.push(reason,'Eval fail by')
            
            if i & 10 == 0:
                self.history.plot('Eval score','Eval time step','Eval average reward','Eval cumulative reward','Eval fail by')
                self.history.save()

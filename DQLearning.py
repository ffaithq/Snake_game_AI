from memory import Transition
from tqdm import tqdm
import torch
from agent import Agent
from snake import Snake
import matplotlib.pyplot as plt

class DQLearning:
    def __init__(self,env,agent,memory,hyperparametrs):
        self.env = env
        self.agent = agent
        self.memory = memory
        self.hyperparametrs = hyperparametrs
        self.history = self.init_history()
    

    def init_history(self):
        return {
            'score':[],
            'time_step':[],
            'average_reward': [],
            'cumulative_reward': [],
            'average_loss': [],
            'score_eval':[],
            'time_step_eval':[],
            'average_reward_eval': [],
            'cumulative_reward_eval': [],
            'reason':[]
        }
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
                state = torch.tensor(state,dtype=torch.float).unsqueeze(0)
                action = self.agent.action(self.hyperparametrs['epsilon'],state)


                done,reward,info = self.env.step(action=action.item())
                cumulative_reward += reward

                reward = torch.tensor([reward],dtype=torch.float)
                
                if done:
                    new_state = None
                else:
                    new_state = torch.tensor(self.env.get_state(),dtype=torch.float).unsqueeze(0)

                self.memory.push(state,action,new_state,reward)

                state = new_state
                batch = self.get_samples()
                if batch:
                    loss += self.agent.optimize_model(batch,self.hyperparametrs['gamma'],criterion)
                    self.agent.update_model(self.hyperparametrs['alpha'])

                time_step += 1

            self.history['score'].append(self.env.get_score())
            self.history['average_reward'].append(cumulative_reward/time_step)
            self.history['cumulative_reward'].append(cumulative_reward)
            self.history['time_step'].append(time_step)
            self.history['average_loss'].append(loss/time_step)

            self.agent.save_model(path_target,path_policy)

            if i % 100 == 0:
                self.save_plot()

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
                state = torch.tensor(state,dtype=torch.float).unsqueeze(0)
                action = self.agent.action(self.hyperparametrs['epsilon'],state,False)
                done,reward,reason = self.env.step(action=action.item())
                
                if done:
                    new_state = None
                else:
                    new_state = torch.tensor(self.env.get_state(),dtype=torch.float).unsqueeze(0)

                state = new_state

                cumulative_reward += reward
                time_step += 1
                
            self.history['score_eval'].append(self.env.get_score())
            self.history['average_reward_eval'].append(cumulative_reward/time_step)
            self.history['cumulative_reward_eval'].append(cumulative_reward)
            self.history['time_step_eval'].append(time_step)

            self.history['reason'].append(reason)
        self.save_plot(mode='_eval')

    def save_plot(self,mode = ''):       
        plt.figure()
        plt.plot(self.history['score' + mode])
        plt.title(f"Score{mode}")
        plt.savefig(f'plots/Score{mode}.png')
        plt.close()

        plt.figure()
        plt.plot(self.history['average_reward' + mode])
        plt.title(f"Average reward{mode}")
        plt.savefig(f'plots/average_reward{mode}.png')
        plt.close()

        plt.figure()
        plt.plot(self.history['cumulative_reward' + mode])
        plt.title(f"Cumulative reward{mode}")
        plt.savefig(f'plots/cumulative_reward{mode}.png')
        plt.close()

        plt.figure()
        plt.plot(self.history['time_step' + mode])
        plt.title(f"Time step{mode}")
        plt.savefig(f'plots/time_step{mode}.png')
        plt.close()

        if mode == '':
                
            plt.figure()
            plt.plot(self.history['average_loss'])
            plt.title(f"Average loss")
            plt.savefig(f'plots/average_loss.png')
            plt.close()

        else:
            plt.figure()
            plt.plot(self.history['reason'])
            plt.title(f"Reason")
            plt.savefig(f'plots/reason.png')
            plt.close()

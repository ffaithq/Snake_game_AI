from snake import Snake
from network import DQN
from DQLearning import DQLearning
from agent import Agent
from memory import Memory
from torch.nn import SmoothL1Loss
from history import Tracker

hyperparametrs = {
    'learning rate':0.005,
    'epsilon': 0.2,
    'batch_size': 256,
    'alpha': 0.05,
    'gamma': 0.9

}


agent = Agent(DQN,12,4,'AdamW',hyperparametrs['learning rate'])
env = Snake(False)
memory = Memory(1000)
tracker = Tracker('./log',hyperparametrs,'Train score',
                        'Train time step',
                        'Train average reward',
                        'Train cumulative reward',
                        'Eval score',
                        'Eval time step',
                        'Eval average reward',
                        'Eval cumulative reward',
                        'Eval fail by')
learning = DQLearning(env,agent,memory,hyperparametrs,tracker)


#learning.train(1000,SmoothL1Loss(),'models\\target\\target.pth','models\\policy\\policy.pth')
learning.eval(100,'models\\target\\target.pth')




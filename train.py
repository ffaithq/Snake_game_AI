from snake import Snake
from network import DQN
from DQLearning import DQLearning
from agent import Agent
from memory import Memory
from torch.nn import SmoothL1Loss

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
learning = DQLearning(env,agent,memory,hyperparametrs)
learning.train(1000,SmoothL1Loss(),'models/target/target.pth','models/policy/policy.pth')
learning.eval(100,'models/target/target.pth')



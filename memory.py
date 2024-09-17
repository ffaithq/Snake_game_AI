from collections import deque,namedtuple
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class Memory(object):
    def __init__(self,length):
        self.memory = deque(maxlen=length)

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def push(self,*args):
        self.memory.append(Transition(*args))

import matplotlib.pyplot as plt
import json
import os
import random


class Tracker:
    def __init__(self,path,hyperparametrs,*args):
        self.parametrs = hyperparametrs
        self.history = {}
        self.path = path
        self.name = str(random.randint(1,1000))
        for k in args:
            self.history[k] = []
        
        os.makedirs(f'{self.path}/{self.name}',exist_ok=True)
        os.makedirs(f'{self.path}/{self.name}/plots', exist_ok=True)
        

    def reset(self):
        for k in self.history:
            self.history[k] = []

    def show(self):
        for k in self.history:
            print(k)

    def get_keys(self):
        return self.history.keys()

    def push(self,value,keys:str):
        self.history[keys].append(value)

    def save(self):
        with open(f'{self.path}/{self.name}/history.json', "w") as outfile: 
            json.dump({
                'hyperparametrs':self.parametrs,
                'history':self.history
            }, outfile)

    def plot(self,*args):
        for key in args:
            plt.figure()
            plt.plot(self.history[key])
            plt.title(f"{key}")
            plt.savefig(f'{self.path}/{self.name}/plots/{key}.png')
            plt.close()



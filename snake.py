import pygame 
import random
import time
import sys
from math import floor
from utils.iou import IoU,box_iou
from collections import namedtuple
import torch

PT = namedtuple('Point','x,y')


WIDTH = 500
HEIGHT = 600
VELOCITY = 800
FPS = 16
EPSILON = 0.00

SIZE = 20


class Snake:
    def __init__(self,render = False) -> None:
        self.head = None
        self.snake = None
        self.direction = None
        self.fruit = None
        self.velocity = VELOCITY
        self.score = 0
        self.render_type = render
        self.objects = torch.zeros(size=(2,4))
        if render:
            self.render()

    def add_target(self):
        self.objects[0] = torch.tensor([self.fruit.x,self.fruit.y,self.fruit.x + SIZE,self.fruit.y + SIZE])

    def add_snake(self):
        to_cat = torch.tensor([[self.snake[0].x,self.snake[0].y,self.snake[0].x + SIZE, self.snake[0].y + SIZE]])
        for part in self.snake[1:]:
            to_cat = torch.cat((to_cat,torch.tensor([[part.x,part.y,part.x + SIZE,part.y + SIZE]])))

        self.objects = torch.cat((self.objects[0].unsqueeze(0),to_cat))

    def render(self):
            self.render_type = True
            pygame.init()
            self.font = pygame.font.Font('utils/Tahoma.ttf', 25)
            self.click = pygame.time.Clock()
            self.SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

    def reset(self):
        self.head = PT(random.randint(200,300),random.randint(200,300))
        self.snake = [self.head]
        self.direction = random.randint(0,3)
        self.score = 0
        self.fruit = None
        self.objects = torch.zeros(size=(1,4))


    def place_fruit(self):
        self.fruit = PT(random.randint(0,WIDTH-20),random.randint(0,HEIGHT-20-100))


    def draw(self):
        self.click.tick(FPS)
        self.SCREEN.fill('black')
        pygame.draw.rect(self.SCREEN,(50,50,50),(0,0,WIDTH,HEIGHT-100))
        if self.fruit is not None:
            pygame.draw.rect(self.SCREEN,'red',(self.fruit.x,self.fruit.y,SIZE,SIZE))
        for pt in self.snake:
            pygame.draw.rect(self.SCREEN,'green',(pt.x,pt.y,SIZE,SIZE))

        text = self.font.render("Score: " + str(self.score), True, (255,255,255))

        self.SCREEN.blit(text, [50, 525])

        pygame.display.flip()


    def move(self,current,dt):
        '''
                if self.direction == 0:
            return current.x - self.velocity*dt, current.y

        if self.direction == 1:
            return current.x, current.y - self.velocity*dt

        if self.direction == 2:
            return current.x + self.velocity*dt, current.y
            
        if self.direction == 3:
            return current.x, current.y + self.velocity*dt
        '''

        


        if self.direction == 0:
            return current.x - 20, current.y

        if self.direction == 1:
            return current.x, current.y - 20

        if self.direction == 2:
            return current.x + 20, current.y
            
        if self.direction == 3:
            return current.x, current.y + 20

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_a:
                    return 0
                if event.key == pygame.K_w:     
                    return 1
                if event.key == pygame.K_d:    
                    return 2
                if event.key == pygame.K_s:                    
                    return 3
                      

        return self.direction
    def is_collision(self):
        if self.snake[0].x >= WIDTH - SIZE or self.snake[0].x <= 0:
            return True, 1
        if self.snake[0].y >= HEIGHT - 100 - SIZE or self.snake[0].y <= 0:
            return True, 1
        
        return self.snake[0] in self.snake[1:], 2

    def will_collision(self,x,y):
        if x >= WIDTH - SIZE or x <= 0:
            return True
        if y >= HEIGHT - 100 - SIZE or y <= 0:
            return True
        
        return PT(x,y) in self.snake[1:]

    def step(self,dt = 0.01,action = None):
        R = 0
        if self.fruit is None:
            self.place_fruit()
            self.add_target()

        if action is None:
            self.direction = self.act()
        else:
            self.direction = action
        self.add_snake()
        new_x,new_y = self.move(self.snake[0],dt)
        self.snake.insert(0,PT(int(new_x),int(new_y)))
        colision_matrix = box_iou(self.objects,self.objects)
        if colision_matrix[1,0]:
            self.fruit = None
            self.score += 1
            R = 10
        else:
            self.snake.pop()


        colision, info = self.is_collision()
        if colision:
            return True,-10,info
        if self.render_type:
            self.draw() 
        return False,R,0

    def play(self,action = None):

        Done = False
        PREV_TIME = time.time()
        self.reset()

        while not Done:

            if self.render_type:
                NOW = time.time()
                dt = NOW - PREV_TIME
                PREV_TIME = NOW

            Done,R,info = self.step(0.01)
            print(self.snake)

    def get_state(self):
        error = 20
        state = []
        if self.fruit is None:
            state += [
                False,
                False,
                False,
                False
        ]
        else:
            state += [
                self.fruit.x+error < self.snake[0].x or self.fruit.x-error < self.snake[0].x,
                self.fruit.y+error < self.snake[0].y or self.fruit.y-error < self.snake[0].y,
                self.fruit.x+error > self.snake[0].x or self.fruit.x-error > self.snake[0].x,
                self.fruit.y+error > self.snake[0].y or self.fruit.y-error > self.snake[0].y,

            ]
        state += [
                self.will_collision(self.snake[0].x - error,self.snake[0].y),
                self.will_collision(self.snake[0].x,self.snake[0].y - error),
                self.will_collision(self.snake[0].x + error,self.snake[0].y),
                self.will_collision(self.snake[0].x,self.snake[0].y + error)
        ]


        if len(self.snake) > 1:
            state += [
                self.snake[0].x > self.snake[1].x,
                self.snake[0].y > self.snake[1].y,
                self.snake[0].x < self.snake[1].x,
                self.snake[0].y < self.snake[1].y,
            ]
        else:
            state += [
                False,
                False,
                False,
                False
            ]
        return state
    def get_score(self):
        return self.score


#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys
import os
import random
import numpy as np

import json
import tensorflow as tf
from keras import initializers
from keras.initializers import normal, identity
#from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam


CONFIG = 'nothreshold'
ACTIONS = 4 # number of valid actions
GAMMA = 0.90 # decay rate of past observations

# about gamma
# gamma is the ratio of how important a current actions rewards are vs the long term projected rewards
# in this maze game where the rewards are often it seems to be better to have gamma set lower (0.80) as opposed to high (0.99) 
# i also note that the end q scores for a reward of 1 tend to be 99 and for reward of -0.1 tend to be -9 (reward/GAMMA).

# with gamma 0.8 however it seems to be stuck within the local area


EXPLORE_STEPS = 1000000.0  # frames over which to anneal epsilon
EXPLORE_INITIAL = 0.1      # starting value of epsilon
EXPLORE_FINAL   = 0.03     # final value of epsilon 

INFORMED_RANDOM   = 0.5      # chance it does a informed random movement

HISTORY_DEPTH = 500
BATCH = 64
LEARNING_RATE = 1e-4

PIT = 0
TREASURE = 1
LAYERS = 2

WIDTH=11
HEIGHT=11

EAST  = 0
SOUTH = 1
WEST  = 2
NORTH = 3

class RobotWorld:
    def __init__(self, width=81, height=51):
        if width == 4 and height == 4:
            # special case game shape 
            self.width  = width 
            self.height = height
        else:
            # Only odd shapes - due to maze builder algo        
            self.width = (width // 2) * 2 + 1
            self.height = (height // 2) * 2 + 1

    def render(self):
        out = ""
        for y in range(self.height):
            for x in range(self.width):
                if x == self.bot_x and y == self.bot_y:
                    if self.world[PIT,y,x] == 1:
                        out += "X"
                    else:
                        out += "@"
                elif self.world[TREASURE,y,x] == 1:
                    out += "."
                elif self.world[PIT,y,x] == 1:
                    out += "#"
                else:
                    out += " "
            out += "\n"

        return out

    def build_outer_walls(self):
        self.world    = np.zeros((LAYERS,self.height,self.width))

        # Fill borders
        self.world[PIT, 0, :] = self.world[PIT, -1, :] = 1
        self.world[PIT, :, 0] = self.world[PIT, :, -1] = 1

    def build_random_walls(self):
        self.world    = np.zeros((LAYERS,self.height,self.width))

        # Fill borders
        self.world[PIT, 0, :] = self.world[PIT, -1, :] = 1
        self.world[PIT, :, 0] = self.world[PIT, :, -1] = 1

        count = int(self.width * self.height * 0.15)

        for z in range(count):
            x         = np.random.random_integers(1, self.width-2)
            y         = np.random.random_integers(1, self.height-2)

            self.world[PIT,y,x] = 1

    # from https://en.wikipedia.org/wiki/Maze_generation_algorithm
    def build_maze(self, complexity=.75, density=.75):
        shape = (self.height, self.width)

        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))

        # Build actual maze
        Z = np.zeros(shape)

        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1

        # Make aisles
        for i in range(density):
            x = np.random.random_integers(0, shape[1] // 2) * 2
            y = np.random.random_integers(0, shape[0] // 2) * 2

            Z[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_,x_ = neighbours[np.random.random_integers(0, len(neighbours) - 1)]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        self.world    = np.zeros((LAYERS,self.height,self.width))
        self.world[PIT,:,:] = Z

    def treasure_fill_all(self):
        self.world[TREASURE, :, :] = 1 - self.world[PIT, :, :]

    def one_treasure(self):
        while self.world[TREASURE, :, :].sum() == -1:   # the bot is -1..
            x         = np.random.random_integers(1, self.width-2)
            y         = np.random.random_integers(1, self.height-2)

            if self.world[PIT, y,x] == 0:               
                self.world[TREASURE, y, x] = 1

    def bot_random(self):
        self.bot_x         = np.random.random_integers(1, self.width-2)
        self.bot_y         = np.random.random_integers(1, self.height-2)

        # insert bot
        self.world[TREASURE, self.bot_y,self.bot_x] = -1
        self.world[PIT     , self.bot_y,self.bot_x] = -1        


    def find_in_room_task(self):
        self.task = "find in room"
        self.build_outer_walls()
        self.bot_random()
        self.one_treasure()

    def explore_room_task(self):
        self.task = "explore room"
        self.build_outer_walls()
        self.treasure_fill_all()
        self.bot_random()

    def find_in_choas_task(self):
        self.task = "find in choas"
        self.build_random_walls()
        self.bot_random()
        self.one_treasure()

    def explore_choas_task(self):
        self.task = "find in choas"
        self.build_random_walls()
        self.treasure_fill_all()
        self.bot_random()

    def find_in_maze_task(self):
        self.task = "find in maze"

        self.build_maze()
        self.bot_random()
        self.one_treasure()

    def explore_all_maze_task(self):
        self.task = "explore maze"

        self.build_maze()
        self.treasure_fill_all()
        self.bot_random()

    def random_task(self):
        task =  np.random.random_integers(5)
        if task == 0:
            self.explore_all_maze_task()
        elif task == 1:
            self.find_in_maze_task()

        elif task == 2:
            self.explore_choas_task()
        elif task == 3:
            self.find_in_choas_task()

        elif task == 4:
            self.explore_room_task()
        elif task == 5:
            self.find_in_room_task()

        else:
            print("task is unknown", task)

    def start(self):  
        # tester worlds.
        self.random_task()        

        self.score         = 0
        self.max_score     = self.world[TREASURE, :, :].sum()  + 1 

    def move(self, move_dir):
        new_x = self.bot_x 
        new_y = self.bot_y 

        if move_dir == EAST:  # 0
            new_x += 1
        if move_dir == SOUTH:  # south
            new_y += 1
        if move_dir == WEST:  # west
            new_x -= 1
        if move_dir == NORTH:  # north
            new_y -= 1

        # implicit PITs..
        new_x = min(max(new_x,0), self.width-1)
        new_y = min(max(new_y,0), self.height-1)

        terminate = False  # still playing
        reward = -0.1      # movement cost...

        # remove bot
        self.world[PIT,    self.bot_y,self.bot_x] =  0
        self.world[TREASURE,self.bot_y,self.bot_x] =  0

        if self.world[TREASURE, new_y, new_x] == 1:
            # self.world[new_y, new_x, TREASURE] = 0
            self.world[TREASURE, new_y, new_x] = 0
            reward = 1.0

            if self.world[TREASURE,:,:].sum() <= 0: 
                terminate = True

        if self.world[PIT, new_y, new_x] == 1:
            # dead ..
            reward = -1.0
            terminate = True

        if self.score <= -10.0:
            # infinite game??
            terminate = True

        self.bot_x = new_x
        self.bot_y = new_y

        # insert bot
        if not terminate:
            self.world[PIT, self.bot_y,self.bot_x] = -1
        self.world[TREASURE, self.bot_y,self.bot_x] = -1

        self.score += reward
        return reward, terminate, self.score

##################### BOT features ##########################

class BotVision:
    # vision wrapper to allow model to view world relative to bot.. (allows network to scale)
    def __init__(self, game, eye):
        self.game = game
        self.eye  = eye
        self.width  = 2*eye + 1
        self.height = 2*eye + 1

    def view(self):
        eye_width = 2*self.eye + 1 
        vision = np.zeros((LAYERS,self.width,self.height))
        vision[PIT,:,:] = 1

        # distances to the worlds wall or vision wall which ever comes first
        wlx = min(self.eye,self.game.bot_x)
        wly = min(self.eye,self.game.bot_y)
        whx = min(self.eye,self.game.width  - self.game.bot_x - 1)
        why = min(self.eye,self.game.height - self.game.bot_y - 1)

        glx = self.game.bot_x - wlx
        gly = self.game.bot_y - wly
        ghx = self.game.bot_x + whx + 1
        ghy = self.game.bot_y + why + 1

        vlx = self.eye - wlx
        vly = self.eye - wly
        vhx = self.eye + whx + 1
        vhy = self.eye + why + 1
        
        vision[PIT     , vly:vhy, vlx:vhx] = self.game.world[PIT     , gly:ghy, glx:ghx]
        vision[TREASURE, vly:vhy, vlx:vhx] = self.game.world[TREASURE, gly:ghy, glx:ghx]        

        return vision
    
    def apply(self):
        return self.view()

class Model1:
    def __init__(self,width,height):
        self.filebase = "model_size" + str(width) + "x" + str(height) + "/model"

        path = os.path.dirname(self.filebase)
        if not os.path.isdir(path):
            os.makedirs(path)

        self.width = width
        self.height = height
            
    def build(self):
        print("############### BUILDING MODEL ####################")
        model = Sequential()
        model.add(InputLayer(input_shape=(LAYERS,self.height,self.width)))
        
        if self.width > 5 or self.height > 5:   
            model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same')) 
            model.add(Activation('relu'))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(ACTIONS))

        filename = self.filebase + ".hd5"
        if os.path.isfile(filename):
            print ("loading prior weights")
            model.load_weights(filename)
       
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)

        self.model = model
        print("###############     DONE      ####################")

    def save(self,step):
        filename = self.filebase + "_step" + str(step) + ".hd5"
        self.model.save_weights(filename, overwrite=True)

        filename = self.filebase + ".hd5"
        self.model.save_weights(filename, overwrite=True)

        #filename = self.filebase + ".json"
        #with open(filename, "w") as outfile:
        #    json.dump(self.model.to_json(), outfile)

    def prepare_one(self,state):
        # reshape into a batch of one...
        return state.reshape((1,) + state.shape) 

    def prepare_batch(self,state):
        # the stat is a loose batch of np.arrays ie a np.array(dtype=object)
        return np.vstack(state).reshape(state.shape + state[0].shape)

    def predict(self,state):
        return self.model.predict(state)

    def train(self,state,targets):
        return self.model.train_on_batch(state, targets)


################## RL trainer #############################
class HistoryStack:
    def __init__(self, depth):
        self.depth = depth
        self.last_idx = 0
        self.tail_idx = 0

        # more efficent?? i think but over specific...
        self.state      = np.empty([depth], dtype=object)
        #self.state      = np.zeros((REPLAY_MEMORY, 
        #                                LAYERS, 
        #                                game.height, 
        #                                game.width), 
        #                                dtype=np.int)  # The world state of before the action
        self.action_idx = np.zeros((depth), dtype=np.int)                   # The action index taken
        self.reward     = np.zeros((depth), dtype=np.float)                 # The instant reward for taking the action
        self.terminal   = np.zeros((depth), dtype=bool)                     # the terminal state of the action sequance

    def push_back(self, state, action_index, reward, terminal):
        self.state[self.tail_idx] =      state
        self.reward[self.tail_idx] =     reward
        self.terminal[self.tail_idx] =   terminal
        self.action_idx[self.tail_idx] = action_index

        self.last_idx = self.tail_idx
        self.tail_idx = (self.tail_idx + 1) % self.depth

    def batch_select(self, count):
        # idx of selected batch items 
        # - Assumes more history that btach size (cause we delay until then anyway)
        # - Also there is a special case issue with last executed state.. its "state t+1" 
        #   has not yet been inserted in the history array so we wont use it

        batch_idx_t = np.arange(self.depth)

        # if not self.terminal[self.tail_idx]:
        batch_idx_t = batch_idx_t[batch_idx_t != self.last_idx]  # remove the tail (its next state is incomplete!)

        # shuffle the idx and make the selection
        np.random.shuffle(batch_idx_t)
        batch_idx_t = batch_idx_t[0:count]
        batch_idx_t1 = np.mod(batch_idx_t + 1, self.depth)

        # now formulate the batch.. from the history of plays "experiance replay"
        state_before  = self.state[batch_idx_t]       # The world state of before the action
        state_after   = self.state[batch_idx_t1]      # The world state after the action
        action_idx    = self.action_idx[batch_idx_t]  # The action index taken
        reward        = self.reward[batch_idx_t]      # The instant reward for taking the action
        terminal      = self.terminal[batch_idx_t]    # the terminal state of the action sequance

        return state_before, state_after, action_idx, reward, terminal

def do_train_cycle(history, model):
    # https://en.wikipedia.org/wiki/Q-learning
    # https://www.youtube.com/watch?v=bHeeaXgqVig&index=2&list=PL4uSLeZ-ET3xLlkPVEGw9Bn4Z8Mbp-SQc
    # https://lopespm.github.io/machine_learning/2016/10/06/deep-reinforcement-learning-racing-game.html

    # create a training batch from the history    
    state_before, state_after, action_idx, reward, terminal = history.batch_select(BATCH)

    state_before = model.prepare_batch(state_before)    
    state_after  = model.prepare_batch(state_after)    

    # note the model is expected to predict the array of q values for all actions
    q_before = model.predict(state_before)  # compute current states "q" value (how worthy the state is)
    q_after  = model.predict(state_after)   # compute the next states  "q" value (how worthy the state is)

    # we assume the agent once it is in the next state will take the action giving the max q value  
    q_after_max = np.amax(q_after, axis=1)         

    # now we keep all "q" values for each action the same.. 
    # targets = np.copy(q_before)
    targets = q_before

    # UNLESS we have new info...
    new_q = np.zeros(reward.shape[0], dtype=np.float)

    # if it wasnt a terminal state.. compute the full q learning computation 
    # NOTE - this Q learning function assumes alpha is 1..
    new_q = reward + GAMMA * q_after_max

    # if its a terminal state.. use the terminal reward we found.. dont mix the next sequance into it.. 
    new_q[terminal] = reward[terminal]                
    
    # insert the adjusted values into the correct locations
    for i in range(targets.shape[0]):
        targets[i, action_idx[i]] = new_q[i]

    #print("#################START TRAIN SET##########################")
    #for i in range(state_t.shape[0]):
    #    print(state_t[i,:,:,:])
    #    print(q_t[i,:])
    #    print(targets[i,:])
    #    print(" + + + ")    
    #    if not terminal_t[i]:
    #        print(state_t1[i,:,:,:])
    #        print(q_t1[i,:])
    #        print(q_max_t1[i])
    #    else:
    #        print("..skip..")
    #    print(" + + + ")    
    #    print("reward:", reward_t1[i])
    #    print("terminal:",terminal_t[i])
    #    print("action:",action_idx_t[i])
    #    print("new_q:", new_q[i])
    #    print("------")
    #print("#################END   TRAIN SET##########################")

    loss = model.train(state_before, targets)
    
    return loss

def trainNetwork(game,features,model,args):
    # create the game set it to need restarting
    run_only  = args["run"] 
    verbose   = args["verbose"] 
    final     = args["final"] 

    terminal = True

    # store the previous observations in replay memory (for training)
    history = HistoryStack(HISTORY_DEPTH) 

    step = 0
    while (True):
        if terminal:
            # restart the game 
            game.start()
            terminal = False

            if verbose or final:
                status = "Step:%10d Task:%15s -- START point...\n" %  (step, game.task)

                print(status + game.render())

        # extract the features the bot would see from the game.
        state = features.apply()

        #choose an action
        action_index = 0

        # ACS convert to a lambda for better control
        if step < EXPLORE_STEPS:
            exploration_chance = EXPLORE_INITIAL - step*(EXPLORE_INITIAL - EXPLORE_FINAL) / EXPLORE_STEPS
        else:
            exploration_chance = EXPLORE_FINAL

        action_choice = random.random()
        if not run_only and action_choice <= exploration_chance:
            explored = "random  "
            q = None
            action_index = random.randrange(ACTIONS)
        else:
            #need to reshape, as the model is setup for batching and this is a one shot
            s_t = model.prepare_one(state)
            q = model.predict(s_t)

            if run_only or action_choice > INFORMED_RANDOM:
                explored = "best    "
                action_index = np.argmax(q)
            else:
                explored = "informed"

                q_min = np.min(q)
                q_prob = np.log(q - q_min + 2).ravel()  # +1 to zero +1 of a chance at every action
                q_prob = q_prob / q_prob.sum() 

                action_index = np.random.choice(ACTIONS, p=q_prob)

        #run the selected action and observed the reward etc 
        reward, terminal, total_score = game.move(action_index)

        # store the transition 
        history.push_back(state, action_index, reward, terminal)

        #only train if done observing
        loss = 0
        if not run_only and step > history.depth:
            loss = do_train_cycle(history, model)

            # save progress every few iterations
            if step % 5000 == 0:
                model.save(step/5000)
        
        step = step + 1

        if verbose or (final and terminal) or step % 3000 == 0:

            
            q_str = "" if q is None else np.array_str(q, precision=3)

            status = "Step:%10d Action:%d Reward:% 2.1f Score:% 5.1f End:%d EXPLORE:%.3f %s %.3f Loss %.2f  PRED: %s" %  \
              (step,         \
               action_index, \
               reward,       \
               total_score,  \
               terminal,     \
               exploration_chance, explored, action_choice,  \
               loss,         \
               q_str)

            print(status + "\n" + game.render())

    print("Episode finished!")
    print("************************")

def playGame(args):
    game     = RobotWorld(args["width"], args["height"])
    vision   = BotVision(game, args["eye"])
    
    # model = Model1(game)
    model = Model1(vision.width, vision.height)
    model.build()

    trainNetwork(game, vision, model, args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-x','--width',   type=int)
    parser.add_argument('-y','--height',  type=int)
    parser.add_argument('-e','--eye',    type=int)
    parser.add_argument('-r','--run',     default=False, action="store_true")
    parser.add_argument('-v','--verbose', default=False, action="store_true")
    parser.add_argument('-f','--final',   default=False, action="store_true")  # start finish points
    args = vars(parser.parse_args())

    print(args)

    playGame(args)

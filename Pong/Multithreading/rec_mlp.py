import logging
from NeuralNetwork import NeuralNetwork
import random

class mlp:
    def __init__(self, loadConfig, name):
        logging.basicConfig(filename='log/player_' + str(name) + '.log', level=logging.DEBUG)
        self.timesteps = 100.0
        self.averageval = 0.0
        self.fakediff = 0.0
        self.knn = NeuralNetwork([2,4,1],2)

    def SaveConfig(self,filename):
        #no Return
        self.knn.save(filename)
        print('Configuration saved: ' + filename)
        
    def predict(self,xpos,ypos,mypos):
        #return action  (up:      u
                      #  down:    d
                      #  nothing: n


        print([xpos,ypos])

        pred = self.knn.predict([[xpos,ypos]])

        diff = pred[0] + self.fakediff - mypos

        if diff > 0.1:
            return 'u'
            print('Up!')
        elif diff < -0.1:
            return 'd'
            print('Down!')
        print('Hold position!')
        return 'n'

    def reward_pos(self,error):
        #was good, but show error to center of bat!
        self.knn.reward_pos(self.fakediff)
        self.average('up')
        print('Got positive reward!')

    def reward_neg(self):
        self.average('down')
        print('Got negative reward!')

    def newfakediff(self):
        self.fakediff = (random.random() - 0.5)  * self.averageval


    def average(self,direction):
        if direction == 'more':
            self.averageval += 1.0/self.timesteps
        else:
            self.averageval -= 1.0/self.timesteps

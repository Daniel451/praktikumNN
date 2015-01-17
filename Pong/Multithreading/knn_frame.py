import logging
from NeuralNetwork import NeuralNetwork
import numpy

class mlp:
    def __init__(self, loadConfig, name):
        logging.basicConfig(filename='log/player_' + str(name) + '.log', level=logging.DEBUG)

        self.name = str(name)
        self.timesteps = 100.0
        self.hitratio = 0.5
        self.fakediff = 0.0
        self.newfakediff()
        self.knn = NeuralNetwork([2,4,1],2)

    def saveconfig(self,filename):
        #no Return
        self.knn.save(filename)
        print('Configuration saved: ' + filename)
        
    def predict(self,xpos,ypos,mypos):
        #return action  (up:      u
                      #  down:    d
                      #  nothing: n

        pred = self.knn.predict([[xpos,ypos]])

        diff = pred[0] + self.fakediff - mypos

        if diff > 0.1:
            print('Player ' + self.name +': up!')
            return 'u'
        elif diff < -0.1:
            print('Player ' + self.name +': down!')
            return 'd'
        print('hold position!')
        return 'n'

    def reward_pos(self,error):
        #was good, but show error to center of bat!
        self.knn.reward(self.fakediff)
        if self.hitratio < 1.0:
            self.hitratio += 1.0/self.timesteps
        print('Player ' + self.name + ': got positive reward! Hitratio is now: ' + str(self.hitratio))

    def reward_neg(self):
        if self.hitratio > 1.0:
            self.hitratio -= 1.0/self.timesteps
        print('Player ' + self.name + ': got negative reward! Hitratio is now: ' + str(self.hitratio))

    def newfakediff(self):
        self.fakediff = numpy.random.normal(0.0,1.0/3.0)*(1.0-self.hitratio)
        # Gauss Normalverteilung von etwa -1 - +1 bei  self.hitratio = 0




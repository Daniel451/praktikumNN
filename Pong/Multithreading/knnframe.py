import logging
from NeuralNetwork import NeuralNetwork
import numpy
import os.path
from bcolors import bcolors

class knnframe:
    def __init__(self, loadConfig, name):
        """
        :param loadConfig:
        :param name:
        :return:
        """

        # Logging
        path = 'log_player_' + str(name) + '.log' # logging file path

        # check if logfile exists
        if not os.path.exists(path):
            file = open(path, "w+")
            file.close()

        # logging stuff
        logging.basicConfig(filename=path, level=logging.DEBUG)

        self.name = str(name)
        self.timesteps = 100.0
        self.hitratio = 0.5
        self.fakediff = 0.0
        self.newfakediff()
        self.knn = NeuralNetwork([2,3,1],2)

    def saveconfig(self,filename):
        #no Return
        self.knn.save(filename)
        print('Configuration saved: ' + filename)
        
    def predict(self,xpos,ypos,mypos):
        #return action  (up:      u
                      #  down:    d
                      #  nothing: n

        pred = self.knn.predict([[xpos,ypos]])
        #print( bcolors.FAIL + 'Player ' + self.name + ' predicted: ' + str(pred[0][0]) + ' with sourcedata: ' + str([xpos,ypos]) + bcolors.ENDC)
        logging.debug('predicting...')
        logging.debug(self.knn.debug())


        return float(pred[0][0])

        self.fakediff = 0.0 #TODO seems not to work like this... damn!
        diff = mypos - pred[0][0]



        #print( bcolors.HEADER + 'Player ' + self.name + ' diff: ' + str(diff) + bcolors.ENDC)

        if diff > 0.1:
            #print('Player ' + self.name +': up!')
            return 'd'
        elif diff < -0.1:
            #print('Player ' + self.name +': down!')
            return 'u'
        #print('hold position!')
        return 'n'

    def reward_pos(self,err):
        self.knn.reward(err)
        #self.knn.reward(self.fakediff)
        print('\a')
        # Verhaeltnis von Treffern vom Schläger zu Out's: 0..1
        self.hitratio += 1.0/self.timesteps
        if self.hitratio > 1.0:
            self.hitratio = 0.0

        print( bcolors.OKGREEN + 'Player ' + self.name + ': got positive reward! Hitratio is now: ' + str(self.hitratio) + bcolors.ENDC )
        self.newfakediff()

    def reward_neg(self,err):
        print('Player ' + self.name + ': error is: ' + str(err))
        self.knn.reward(err)
        # Verhaeltnis von Treffern vom Schläger zu Out's: 0..1
        self.hitratio -= 1.0/self.timesteps
        if self.hitratio < 0.0:
            self.hitratio = 0.0

        print( bcolors.OKBLUE + 'Player ' + self.name + ': got negative reward! Hitratio is now: ' + str(self.hitratio) + bcolors.ENDC)
        self.newfakediff()

    def newfakediff(self):
        self.fakediff = numpy.random.normal(0.0,1.0/3.0)*(1.0-self.hitratio)
        # Gauss Normalverteilung von etwa -1 - +1 bei  self.hitratio = 0
        print('Player ' + self.name + ': fakediff is now: ' + str(self.fakediff))




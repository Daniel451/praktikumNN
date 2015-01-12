import numpy as np
import random
import time

class court:
    def __init__(self):
        
        self.x_max = 16.0
        self.y_max = 9.0
        self.alpha_min = 0.5  #als radiant im Einheitskreis! der Winkel der maximalen Spreizung der Richtungsvektoren
        self.initspeed = 0.2
        self.speedstep = 0.05
        self.outputNoiseMax = 0.0
        self.infinite = True
        self.batsize = 1.0 # half of length!
        self.batstep = 0.2 
        
        #### ^^^ Parameter zum ändern ^^^ ####
        
        self.posVec = None
        self.dirVec = None
        self._initVectors()
        self.speed = self.initspeed
        self.bathit = [False, False]
        self.Points = [0, 0]
        self.bat = [self.y_max/2.0 , self.y_max/2.0]
        
        
    def _initVectors(self):
        rad =  (np.pi - 2.0 * self.alpha_min) * 2 * random.random() - ( (np.pi / 2.0 ) - self.alpha_min)
        self.dirVec = np.array([ np.cos(rad) , np.sin(rad) ])
        if random.random() > 0.5:
            self.dirVec[0]= self.dirVec[0] * -1.0
        
        self.posVec = np.array([self.x_max/2.0,self.y_max * random.random()])
    
    def _incrSpeed(self):
        self.initspeed += self.speedstep
    
    def _incrPoints(self,player):
        self.Points[player] += 1
    
    def sensor_X(self): #TODO: Anpassen, input value ist ja nicht 0..16!!!
        return self.posVec[0] + (random.random() - 0.5 ) * self.outputNoiseMax
        
    def sensor_Y(self): #TODO: Anpassen, input value ist ja nicht 0..16!!!
        return self.posVec[1] + (random.random() - 0.5 ) * self.outputNoiseMax
        
    def sensor_bat(self, player): #TODO: Anpassen, input value ist ja nicht 0..16!!!
        return self.bat[player] + (random.random() - 0.5 ) * self.outputNoiseMax
        
        
    def scaled_sensor_x(self):
        return self.sensor_X() / (self.x_max/2.0) - 1.0
        
    def scaled_sensor_y(self):
        return self.sensor_Y() / (self.y_max/2.0) - 1.0 
        
    def scaled_sensor_bat(self, player): 
        return self.sensor_bat(player) / (self.y_max/2.0) - 1.0 
        
    def hitbat(self, player):
        return self.bathit[player]
    
    def getpoints(self, player):
        return self.Points[player]
    
    def tic(self):
        
        self.posVec = self.posVec + self.dirVec
        self.bathit = [False, False]
        # bounce bottom:
        if self.posVec[1] < 0:
            self.posVec[1] = self.posVec[1] * -1.0
            self.dirVec[1] = self.dirVec[1] * -1.0
        
        # bounce top:
        if self.posVec[1] > self.y_max:
            self.posVec[1] = 2 * self.y_max - self.posVec[1]
            self.dirVec[1] = self.dirVec[1] * -1.0
        
        # bounce left:
        if self.posVec[0] < 0:
            
            factor = (0 - self.posVec[0]) / self.dirVec[0]
            poi = self.posVec - (factor * self.dirVec) #point of impact
            if poi[0] > self.bat[0] - self.batsize and poi[0] < self.bat[0] + self.batsize:
                self.bathit[0] = True
            else: 
                self.Points[1] +=1
                
            if self.infinite or self.bathit[0]:
                self.posVec[0] = self.posVec[0] * -1.0
                self.dirVec[0] = self.dirVec[0] * -1.0
        
        # bounce right:
        if self.posVec[0] > self.x_max:
            factor = (self.x_max - self.posVec[0]) / self.dirVec[0]
            poi = self.posVec - (factor * self.dirVec) #point of impact
            if poi[0] > self.bat[1] - self.batsize and poi[0] < self.bat[1] + self.batsize:
                self.bathit[1] = True
            else: 
                self.Points[0] +=1
            
            if self.infinite or self.bathit[1]:
                self.posVec[0] = 2 * self.x_max - self.posVec[0]
                self.dirVec[0] = self.dirVec[0] * -1.0
        
    def move(self, player, action):
        if action == 'd' : #up
            self.bat[player] += self.batstep
            if self.bat[player] > self.y_max:
                self.bat[player] = self.y_max
        if action == 'u': #down
            self.bat[player] -= self.batstep
            if self.bat[player] < 0.0:
                self.bat[player] = 0.0
    
############################# For visu only!: #############################
    def v_getSize(self):
        return [self.x_max,self.y_max]
    def v_getSpeed(self):
        return self.initspeed
    def v_getBatSize(self):
        return self.batsize
    def v_getDirVec(self):
        return self.dirVec
    def v_getPosVec(self):
        return self.posVec
    def v_getbat(self):
        return self.bat
    def v_getPoint(self):
        return self.Points


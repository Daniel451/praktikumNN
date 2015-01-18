import numpy as np
import random



class court:


    def __init__(self):

        # Spielfeld
        self.x_max = 16.0
        self.y_max = 9.0

        # Als radiant im Einheitskreis! der Winkel der maximalen Spreizung der Richtungsvektoren
        self.alpha_min = 0.5

        # Ballgeschwindigkeit
        self.initspeed = 0.2

        # Inkrement zur Geschwindigkeitserhöhung (von initspeed)
        self.speedstep = 0.05

        # Rauschen auf die Ballposition hinzufügen (Faktor)
        self.outputNoiseMax = 0.0

        # Soll der Ball aus dem Spielfeld fliegen können oder ewig hin und her springen?
        self.infinite = False


        self.batsize = 1 # half of length! gesehen auf die y_max!
        self.batstep = 0.3
        
        #### ^^^ Parameter zum ändern ^^^ ####
        
        self.posVec = None
        self.dirVec = None
        self.__initvectors()
        self.speed = self.initspeed
        self._bathit = [False, False]
        self._out = [False, False]
        self.Points = [0, 0]
        self.bat = [self.y_max/2.0 , self.y_max/2.0]
        
        
    def __initvectors(self):
        rad =  (np.pi - 2.0 * self.alpha_min) * 2 * random.random() - ( (np.pi / 2.0 ) - self.alpha_min)
        self.dirVec = np.array([ np.cos(rad) , np.sin(rad) ])
        if random.random() > 0.5:
            self.dirVec[0]= self.dirVec[0] * -1.0
        
        self.posVec = np.array([self.x_max/2.0,self.y_max * random.random()])


    def _incrSpeed(self):
        self.initspeed += self.speedstep


    def _incrPoints(self,player):
        self.Points[player] += 1


    def __sensor_x(self):
        return self.posVec[0] + (random.random() - 0.5 ) * self.outputNoiseMax


    def __sensor_y(self):
        return self.posVec[1] + (random.random() - 0.5 ) * self.outputNoiseMax


    def __sensor_bat(self, player):
        return self.bat[player] + (random.random() - 0.5 ) * self.outputNoiseMax
        
        
    def scaled_sensor_x(self):
        return self.__sensor_x() / (self.x_max/2.0) - 1.0


    def scaled_sensor_y(self):
        return self.__sensor_y() / (self.y_max/2.0) - 1.0


    def scaled_sensor_bat(self, player): 
        return self.__sensor_bat(player) / (self.y_max/2.0) - 1.0


    def hitbat(self, player):
        return self._bathit[player]

    def out(self, player):
        return self._out[player]

    def getpoints(self, player):
        """
        Getter
        :param player: 0 oder 1
        :return: Punktzahl des Spielers
        """
        return self.Points[player]


    def tick(self):
        """

        :return:
        """
        #########################
        ### Initialisierungen ###
        #########################

        # todo: kommentieren
        self.posVec = self.posVec + self.dirVec

        # Hat der Schläger den Ball getroffen?
        # bathit[0] -> linker Schläger
        # bathit[1] -> rechter Schläger
        self._bathit = [False, False]
        self._out = [False, False]
        ###################
        ### Anweisungen ###
        ###################

        # bounce bottom:
        if self.posVec[1] < 0:
            self.posVec[1] = self.posVec[1] * -1.0
            self.dirVec[1] = self.dirVec[1] * -1.0
        
        # bounce top:
        if self.posVec[1] > self.y_max:
            self.posVec[1] = 2 * self.y_max - self.posVec[1]
            self.dirVec[1] = self.dirVec[1] * -1.0
        
        # bounce left
        self.__tickBounceLeft()
        
        # bounce right:
        self.__tickBounceRight()


    def __tickBounceLeft(self):
        """
        Checken, ob der Ball links aus dem Spielfeld fliegt oder vom Schläger getroffen wird

        :type self
        :return: void
        """

        if self.posVec[0] < 0:

            # todo: kommentieren
            factor = (0 - self.posVec[0]) / self.dirVec[0]

            # point of impact
            poi = self.posVec + (factor * self.dirVec)

            print('Left: POI: ' + str(poi))

            # Wenn der Ball über der unteren Kante und unter der oberen Kante des Schlägers
            # auftrifft, so würde der Schläger den Ball treffen
            if (poi[1] > self.bat[0] - self.batsize) and (poi[1] < self.bat[0] + self.batsize):
                self._bathit[0] = True # Schläger hat getroffen
            else:
                self.Points[1] +=1 # Punkte von Spieler 1 (rechts) erhöhen
                self._out[0] = True #Ball ist außerhalb des Spielfelds (gewesen)

            # Ball abprallen lassen, falls:
            # -> Das infinite true ist, also das Spiel endlos dauern soll ohne Zurücksetzen der Ballposition
            # -> Der Schläger den Ball getroffen hat
            if self.infinite or self._bathit[0]:
                self.posVec[0] = self.posVec[0] * -1.0
                self.dirVec[0] = self.dirVec[0] * -1.0
            else:
                self.__initvectors()




    def __tickBounceRight(self):
        """
        Checken, ob der Ball rechts aus dem Spielfeld fliegt oder vom Schläger getroffen wird

        :return: void
        """

        if self.posVec[0] > self.x_max:

            # todo: kommentieren
            factor = (self.x_max - self.posVec[0]) / self.dirVec[0]

            # point of impact
            poi = self.posVec + (factor * self.dirVec)

            print('Right: POI: ' + str(poi))

            # Wenn der Ball über der unteren Kante und unter der oberen Kante des Schlägers
            # auftrifft, so würde der Schläger den Ball treffen
            if poi[1] > self.bat[1] - self.batsize and poi[1] < self.bat[1] + self.batsize:
                self._bathit[1] = True
            else:
                self.Points[0] +=1 # Punkte von Spieler 0 (links) erhöhen
                self._out[1] = True #Ball ist außerhalb des Spielfelds (gewesen)

            # Ball abprallen lassen, falls:
            # -> Das infinite true ist, also das Spiel endlos dauern soll ohne Zurücksetzen der Ballposition
            # -> Der Schläger den Ball getroffen hat
            if self.infinite or self._bathit[1]:
                # 2 Spielfeldlängen - aktuellem X-Betrag ergibt neue X-Position
                self.posVec[0] = 2 * self.x_max - self.posVec[0]
                self.dirVec[0] = self.dirVec[0] * -1.0
            else:
                self.__initvectors()

    def move(self, player, action):
        """
        Bewegt den Schläger eines Spielers

        :param player: 0 oder 1 (Spieler, dessen Schläger bewegt werden soll)
        :param action: "d" oder "u" (Schläger hoch oder runter bewegen)
        :return: void
        """

        # Schläger nach oben bewegen
        if action == 'd' :
            self.bat[player] += self.batstep
            if self.bat[player] > self.y_max: # Korrektur, falls oberer Spielfeldrand erreicht wurde
                self.bat[player] = self.y_max

        # Schläger nach unten bewegen
        if action == 'u':
            self.bat[player] -= self.batstep
            if self.bat[player] < 0.0: # Korrektur, falls unterer Spielfeldrand erreicht wurde
                self.bat[player] = 0.0


    def v_getSize(self):
        """
        visu-getter
        :return: Liste [X,Y] der Spielfeldgröße
        """
        return [self.x_max,self.y_max]


    def v_getSpeed(self):
        """
        visu-getter
        :return: Ballgeschwindigkeit
        """
        return self.initspeed


    def v_getBatSize(self):
        """
        visu-getter
        :return: Schlägerlänge (Größe)
        """
        return self.batsize


    def v_getDirVec(self):
        """
        visu-getter
        :return: Bewegungsvektor
        """
        return self.dirVec


    def v_getPosVec(self):
        """
        visu-getter
        :return: Positionsvektor
        """
        return self.posVec


    def v_getbat(self):
        """
        visu-getter
        :return: Liste [batSpieler0, batSpieler1] -> Position des Schlägermittelpunktes von Spieler 0 / 1
        """
        return self.bat


    def v_getPoint(self):
        """
        visu-getter
        :return: Liste [X,Y] des Punktestundes für Spieler 0 / 1
        """
        return self.Points


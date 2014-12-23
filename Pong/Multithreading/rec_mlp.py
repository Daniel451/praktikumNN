import logging
class mlp:
    def __init__(self, loadConfig, name):
        logging.basicConfig(filename='log/player_' + str(name) + '.log', level=logging.DEBUG)
    def SaveConfig(self,filename):
        #no Return
        print('done ;)')
        
    def predict(self,xpos,ypos,mypos):
        #return action  (up:      u
                      #  down:    d
                      #  nothing: n
        return 'u'
    def reward(self,error):
        #was good, but show error to center of bat!
        print('Thanks!')
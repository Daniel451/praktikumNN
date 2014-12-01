#-*- coding: utf-8 -*-


import numpy



class SARSAloop:

    
    def learn(self, world, action, plot, beta, w, size_motion, size_map, size_x, size_y):

        for iter in xrange(100000):

            world.newRandomStartPosition()
            I = world.get_sensor()
            h = numpy.dot(w, I)
            doAction = action.nextAction(h, beta)    # nächste action bestimmen...
            doAction_vec = numpy.zeros(size_motion)
            doAction_vec[doAction] = 1.0
            val = numpy.dot(w[doAction], I)      # Sichere W von vor der nächsten Action 
            r = 0
            duration = 0
            
            while r == 0:
                
                duration += 1
                
                world.doAction(doAction)
                r = world.get_reward()
                SensorVal = world.get_sensor()
                
                h = numpy.dot(w, SensorVal)
                doAction_tic = action.nextAction(h, beta)
                
                doAction_vec = numpy.zeros(size_motion)
                doAction_vec[doAction] = 1.0
                
                wDotSensorVal = numpy.dot(w[doAction_tic], SensorVal)
                
                if  r == 1.0:  # This is cleaner than defining
                    target = r                                  # target as r + 0.9 * wDotSensorVal,
                else:                                           # because weights now converge.
                    target = 0.9 * wDotSensorVal                # gamma = 0.9
                delta = target - val                            # prediction error
                
                w += 0.3 * (target - val) * numpy.outer(doAction_vec, I)
                
                I[0:size_map] = SensorVal[0:size_map]
                val = wDotSensorVal
                doAction = doAction_tic
            
            print('------------- Needed hops: ' + str(duration) + '-------------')

            if iter%10 == 0:
                plot.plot(w, size_x, size_y) 

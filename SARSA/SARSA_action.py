#-*- coding: utf-8 -*-

# imports
import numpy




class SARSAaction: 


    def nextAction (self, S_from, beta):

        sum = 0.0
        p_i = 0.0
        rnd = numpy.random.random()
        d_r = len (S_from)
        sel = 0

        for i in xrange(d_r):
            sum += numpy.exp (beta * S_from[i])

        S_target = numpy.zeros (d_r)

        for i in xrange(d_r):
            p_i += numpy.exp (beta * S_from[i]) / sum

            if p_i > rnd:
                sel = i
                rnd = 1.1 # out of reach, so the next will not be turned ON

        return sel



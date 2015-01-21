import numpy
import KTimage as KT

class world_bird_chirp:
    """reads the "bird_chirp.pnm" spectrum and returns a frequency amplitude vector for each time step"""
    def __init__(self):
        self.values, self.height, self.width = KT.importimage ("bird_chirp.pnm")
        print "values' shape=", numpy.shape(self.values), " height=", self.height, " width=", self.width
        # values have shape 129*182; now bring them into shape (129,182)
        self.values = numpy.reshape(self.values, (self.height,self.width))
        print "values' shape=", numpy.shape(self.values)
        KT.exporttiles (self.values, self.height, self.width, "/tmp/coco/obs_I_0.pgm") # display
        # transpose this data matrix, so one data point is values[.]
        self.values = numpy.transpose(self.values)
        KT.exporttiles (self.values[0], self.height, 1, "/tmp/coco/obs_J_0.pgm")
        self.t = 0
    def dim(self):
        return (self.height, self.width)
    def newinit(self):
        self.t = 0
    def act(self):
        # world reaction
        self.t += 1
    def sensor(self):
        # return one column of bird_chirp: a 129 long frequency amplitude vector
        return self.values[self.t]

class world_digits:
    """reads the 8x8 image files in digits_alph that display digits and capital letters"""
    def __init__(self):
        digitsalph = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.values = numpy.zeros((len(digitsalph), 8*8))
        for tt in range(len(digitsalph)):
            filename = "digits_alph/digit" + digitsalph[tt] + ".pgm"
            self.values[tt], h, w = KT.importimage (filename)
            if  h != 8 or w != 8:
                print "digits_alph files don't match expected size!"
        self.datumsize = h*w
        self.seqlen = len(digitsalph)
    def dim(self):
        return (self.datumsize, self.seqlen)
    def newinit(self):
        self.t = 0
    def act(self):
        # world reaction
        self.t += 1
    def sensor(self):
        return self.values


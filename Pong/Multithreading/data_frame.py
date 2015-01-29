#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
""" bla bla bla"""   #TODO: Write summary about this file!

__author__ = "Daniel Speck, Florian Kock"
__copyright__ = "Copyright 2014, Praktikum Neuronale Netze"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Daniel Speck, Florian Kock"
__email__ = "2speck@informatik.uni-hamburg.de, 2kock@informatik.uni-hamburg.de"
__status__ = "Development"

class DataFrame:
    def __init__(self,instruction='NULL'):
        self.instruction = instruction
        self.data = {'key':'value'} #TODO: hier habe ich gepfuscht!
    def add(self,identifier, value):
        self.data[identifier] = value
    def getdata(self, identifier):
        return self.data[identifier] 
    def instruction(self):
        return self.instruction
    def debug(self):
        print (self.data)
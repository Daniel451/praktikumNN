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
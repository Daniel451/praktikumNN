#!/usr/bin/env python3.4

import time


class Speedtest:
    """
    class for speedtesting purposes

    usage:
    tester = Speedtest()
    tester.recore("someName") <-- custom name
    tester.record() <-- default name
    tester.printRecords() <-- print all time records
    """

   # datasets
   # datasets is a list of tuples ( name:string, time:float )
    dataset = []

   # seperator
    maxNameSize = 0

   
    def record(self, name="no-name"):
        self.dataset.append((name, time.time()))

        # calculate new maxlength
        if self.maxNameSize < len(name):
            self.maxNameSize = len(name)


    def printRecords(self):
        for i in range (0, len(self.dataset) - 1):
            print( "Zeitunterschied: "
                   + self.returnSeperatedString(self.dataset[i][0])
                   + " <-> "
                   + self.returnSeperatedString(self.dataset[i+1][0])
                   + ": "
                   + str(self.dataset[i+1][1] - self.dataset[i][1])
                   + "s")


    def returnSeperatedString(self, strIn):
        """
        Returns a string which spaces at the end for seperation purposes

        :param strIn: Input string
        :return: str: Output string
        """

        retStr = str(strIn)

        for i in range(len(retStr), self.maxNameSize):
            retStr += " "

        return retStr
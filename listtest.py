

import speedtest as sp


speedrecord = sp.Speedtest()

a = []
b = []
c = []


speedrecord.record("Anfang")

# a fuellen
for i in range(0,10000000):
   a.append(i)

speedrecord.record("a befuellt")

# b berechnen
for i in range(len(a)-1,-1,-1):
   b.append( a[i] * 2 + 1 )

speedrecord.record("b berechnet (a[i])")

# c berechnen
for i in range(0,len(a)):
   c.append( a.pop() * 2 + 1 )

speedrecord.record("c berechnet (a.pop())")

speedrecord.printRecords()

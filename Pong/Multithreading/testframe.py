from data_frame import DataFrame
x = DataFrame('test')

x.add('data' , 1234)

print(x.instruction)
print(x.getdata('data') + 4)


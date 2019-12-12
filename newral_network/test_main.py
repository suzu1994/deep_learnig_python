import numpy as np
import matplotlib.pylab as plt

from utility.filter import *

zahyou = Zahyou()
inp = InputLayer()
middle = MiddleLayer()
out = OutLayer()
result = [[],[],[],[]]

for i in range(20):
    for j in range(20):
        out_result = out.out_layer(middle.middle_layer(inp.input_layer([zahyou.X[i],zahyou.Y[j]])))
        if out_result[0] > out_result[1]:
            result[0].append(zahyou.X[i])
            result[1].append(zahyou.Y[j])
        else:
            result[2].append(zahyou.X[i])
            result[3].append(zahyou.Y[j])

plt.scatter(result[0],result[1],marker = "+")
plt.scatter(result[2],result[3],marker = "o")
plt.show()
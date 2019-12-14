from layer.outputlayer import *
from layer.middlelayer import *
from setting.setting import *
import numpy as np
import matplotlib.pyplot as plt
print(np.arange(0,1,0.1))
input_data = np.arange(0, np.pi * 2, 0.1)
correct_data = np.sin(input_data)
input_data = (input_data - np.pi)/np.pi
n_data = len(correct_data)

middle_layer = MiddleLayer(Setting.wb_width,Setting.n_in,Setting.n_mid)
output_layer = OutputLayer(Setting.wb_width,Setting.n_mid,Setting.n_out)

for i in range(Setting.epoch):
    index_random = np.arange(n_data)
    np.random.shuffle(index_random)

    total_error = 0
    plot_x = []
    plot_y = []
    for idx in index_random:
        x = input_data[idx:idx+1]
        t = correct_data[idx:idx+1]

        middle_layer.forward(x.reshape(1,1))
        output_layer.forward(middle_layer.y)

        output_layer.backfard(t.reshape(1,1))
        middle_layer.backward(output_layer.grad_x)

        middle_layer.update(Setting.eta)
        output_layer.update(Setting.eta)

        if i%Setting.interval == 0:
            y = output_layer.y.reshape(-1)

            total_error += 1.0/2.0*np.sum(np.square(y - t))
            plot_x.append(x)
            plot_y.append(y)
    
    if i%Setting.interval == 0:

        plt.plot(input_data,correct_data,linestyle="dashed")
        plt.scatter(plot_x,plot_y,marker="+")
        plt.show()
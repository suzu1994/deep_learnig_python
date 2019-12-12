import numpy as np
class Zahyou:
    def __init__(self):
        self.X = np.arange(-1.0,1.0,0.1)
        self.Y = np.arange(-1.0,1.0,0.1)

class MiddleLayer:
    w = np.array([[1.0,2.0],[2.0,3.0]])
    b = np.array([0.3,-0.3])
    def middle_layer(self,x):
        u = np.dot(x,MiddleLayer.w) + MiddleLayer.b
        return 1 / (1 + np.exp(-u))

class OutLayer:
    w = np.array([[-1.0,1.0],[1.0,-1.0]])
    b = np.array([0.4,0.1])
    def out_layer(self,x):
        u = np.dot(x,OutLayer.w) + OutLayer.b
        return np.exp(u) / np.sum(np.exp(u))
    
class InputLayer:
    def input_layer(self,x):
        return np.array(x)
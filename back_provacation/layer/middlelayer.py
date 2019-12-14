import numpy as np
class MiddleLayer:
    def __init__(self, wb, n_upper, n):
        self.w = wb * np.random.randn(n_upper, n)
        self.b = wb * np.random.randn(n)
    
    def forward(self, x):
        self.x = x
        u = np.dot(x,self.w) + self.b
        self.y = 1/(1 + np.exp(-u))

    def backward(self, grad_y):
        delta = grad_y * (1 - self.y) * self.y

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis = 0)
        self.grad_x = np.dot(delta, self.w.T)
    
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
    

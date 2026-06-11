import torch
from micro import Value
import random
import numpy

class Neuron:
    def __init__(self,nin):
        self.w = [(Value(random.uniform(-1,1))) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w ,x)), self.b)
        out = act.tanh()
        return out
    
x = [2.0, 3.0]
n = Neuron(2)
print(n(x))

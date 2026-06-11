import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3*x**2 - 5*x + 9
# Finding out the derivaties with respect to each a,b and c
h = 0.00001
a = 2.0
b = -3.0
c = 10
d1 = a*b + c
c += h
d2 = a*b + c
# 

class Value:
    def __init__(self,data,_children=(), _op = '', label = ''):
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(
            self.data + other.data,
            (self,other), "+"
        )
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    
    def __sub__(self,other):
        return (self.data + (-other.data))
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(
            self.data * other.data,
            (self,other), "*"
        )
        def _backward():
            self.grad += other.grad*out.grad
            other.grad += self.grad*out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return (self*other)
    
    def __pow__(self,other):
        assert isinstance(other, (int, float)), \
        "only supporting int/float powers for now"
        out = Value(
            self.data ** other,
            (self,),
            f"**{other}"
        )
        def backward():
            self.grad += other*(self.data*(other -1))*out.grad
        out._backward = self._backward
        return out
    def __truediv__(self, other):
        return((self.data)/other.data)


    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),"exp")
        def backward():
            self.grad = out.data*out.grad
        out._backward = self._backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,))
        def _backward():
               self.grad += (1 - t**2)*out.grad
        out._backward = _backward
        return out 
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
    



a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c

# plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2)))
# plt.show()


b = Value(6.7 , label = 'b')
x1 = Value(1.0, label ='x1')
x2 = Value(0.0, label ='x2')
w1 = Value(-3.0, label ='w1')
w2 = Value(4.0, label ='w2')
x1w1x2w2 = x1*w1 + x2*w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh()
a = Value(3.0)
b = Value(2.0)
print (a - b)

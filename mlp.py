from engine import Tensor
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(random.uniform(-1, 1))
    
    def forward(self, x):
        res = [x1 * w1 for x1,w1 in zip(x,self.w)]
        return sum(res) + self.b
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, neuronNum):
        self.neurons = [Neuron(nin=nin) for _ in range(neuronNum)]

    # 每一层的每一个神经元接受相同的输入，然后输出每一个神经元的输出
    def forward(self, x):
        out = [neuron.forward(x) for neuron in self.neurons]
        return out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class mlp:
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
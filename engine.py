import math


class Tensor:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._prev = set(_children) # 记录前置子节点
        self._backward = lambda: None #每一个节点需要知道计算自己的反向方法
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        o = Tensor(self.data + other.data, (self,other))
        
        # 加法的局部梯度为 1
        def backward():
            self.grad +=  1 * o.grad # 使用 += 因为节点可能有多个父节点，不同父节点反向传递的梯度需要相加
            other.grad += 1 * o.grad
        o._backward = backward

        return o

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other) 
        o = Tensor(self.data * other.data, (self,other))

        def backward():
            self.grad += other.data * o.grad
            other.grad += self.data * o.grad
        o._backward = backward

        return o

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        o = Tensor(self.data ** other, (self,))

        def backward():
            self.grad += other * (self.data **(other - 1)) * o.grad
        o._backward = backward
        
        return o
    
    def tanh(self):
        o = Tensor(math.tanh(self.data),(self,))

        def backward():
            self.grad += (1 - math.tanh(self.data)**2 ) * o.grad
        
        o._backward = backward

        return o

    # -- 下方的所有运算都通过上方的基础运算实现 -- #

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (- other)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return Tensor(other) - self
    
    # -- 除法运算 通过乘以倒数实现 -- #
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    # 反向传播 
    def backward(self):
        # 获取拓扑顺序
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1  # 设置起点的梯度为 1

        # 反向拓扑计算
        for tensor in reversed(topo):
            tensor._backward()
    
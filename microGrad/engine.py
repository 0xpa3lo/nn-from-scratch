import math

class Value:
    # attributes data, operation, children, label, grad
    def __init__(self, data, _children=(), _op='', label=None):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self.label = label
        self._op = _op
    
    # wrapper 
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # basic math operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  
        out = Value(self.data + other.data, _children=(self, other), _op="+")
        
        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), _op="/")
        def backward():
            self.grad += (1/other.data) * out.grad
            other.grad += (1/self.data) * out.grad
        
        out._backward = backward
        return out
    
    def __rtruediv__(self, other):
        return Value(other) / self
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
    
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), _op="log")

        def backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op="tanh")

        def backward():
            self.grad += (1 - t**2)  * out.grad
        
        out._backward = backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    # O(z) = 1 / 1 + e^-z
    def sigmoid(self):
        out = Value(1.0 / (1.0 + math.exp(-self.data)), (self,), _op='sigmoid')
       # print(f"Sigmoid output: {out.data}")
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad

        out._backward = _backward
        return out
    
    # topological sort 
    '''
    visited         topo
    {}               {}
    {D}              {}            
    {D,C}            {}
    {D,C}            {D}
    {D,C}            {D,C}   
    '''
    def backward(self):
        sorted = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
            sorted.append(node)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(sorted):
            node._backward()
    
    
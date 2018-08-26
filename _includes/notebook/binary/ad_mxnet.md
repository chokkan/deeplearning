

```python
import mxnet as mx
from mxnet import nd, autograd, gluon

x = nd.array([1., 1., 1.])
w = nd.array([1.0, 1.0, -1.5])
w.attach_grad()

with autograd.record():
    loss = -nd.dot(x, w).sigmoid().log()
loss.backward()
print(loss)
print(w.grad)
```

    
    [0.47407696]
    <NDArray 1 @cpu(0)>
    
    [-0.37754065 -0.37754065 -0.37754065]
    <NDArray 3 @cpu(0)>


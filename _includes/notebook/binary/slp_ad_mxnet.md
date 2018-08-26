

```python
import mxnet as mx
from mxnet import nd, autograd

# Training data for NAND.
x = nd.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = nd.array([[1], [1], [1], [0]])
w = nd.random.normal(0, 1, shape=(3, 1))
w.attach_grad()

eta = 0.5
for t in range(100):
    with autograd.record():
        # y_pred = \sigma(x \cdot w).
        y_pred = nd.dot(x, w).sigmoid()
        ll = y * y_pred + (1 - y) * (1 - y_pred)
        loss = -ll.log().sum()      # The loss value.
        #print(t, loss)
    loss.backward()                 # Compute the gradients of the loss.
    w -= eta * w.grad               # Update weights using SGD.
```


```python
w
```




    
    [[-4.2020216]
     [-4.20314  ]
     [ 6.4963117]]
    <NDArray 3x1 @cpu(0)>




```python
nd.dot(x, w).sigmoid()
```




    
    [[0.9984933 ]
     [0.90831   ]
     [0.90840304]
     [0.12911019]]
    <NDArray 4x1 @cpu(0)>



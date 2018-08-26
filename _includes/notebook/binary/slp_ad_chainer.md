

```python
import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F

dtype = np.float32

# Training data for NAND
x = Variable(np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=dtype))
y = Variable(np.array([[1], [1], [1], [0]], dtype=dtype))
w = Variable(np.random.rand(3, 1).astype(dtype=dtype), requires_grad=True)

eta = 0.5
for t in range(100):
    # y_pred = \sigma(x \cdot w)
    y_pred = F.sigmoid(F.matmul(x, w))
    ll = y * y_pred + (1 - y) * (1 - y_pred)
    loss = -F.sum(F.log(ll))    # The loss value.
    #print(t, loss)
    loss.backward()             # Compute the gradients of the loss.

    with chainer.no_backprop_mode():
        w -= eta * w.grad       # Update weights using SGD.
        w.cleargrad()           # Clear the gradients for the next iteration.
```


```python
w
```




    variable([[-4.238654 ],
              [-4.2391624],
              [ 6.550305 ]])




```python
F.sigmoid(F.matmul(x, w))
```




    variable([[0.99857235],
              [0.90979564],
              [0.90983737],
              [0.12702626]])



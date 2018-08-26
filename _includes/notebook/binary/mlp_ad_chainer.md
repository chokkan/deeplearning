

```python
import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F

dtype = np.float32

# Training data for XOR.
x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=dtype)
y = np.array([[0], [1], [1], [0]],dtype=dtype)
w1 = Variable(np.random.randn(3, 2).astype(dtype),requires_grad=True)
w2 = Variable(np.random.randn(2, 1).astype(dtype),requires_grad=True)
b2 = Variable(np.random.randn(1).astype(dtype), requires_grad=True)

eta = 0.5
for t in range(1000):
    # y_pred = \sigma(w_2 \cdot \sigma(x \cdot w_1) + b_2)
    y_pred = F.sigmoid(F.bias(F.matmul(F.sigmoid(F.matmul(x, w1)), w2), b2))
    ll = y * y_pred + (1 - y) * (1 - y_pred)
    loss = -F.sum(F.log(ll))
    #print(t, loss.data)
    loss.backward()
    with chainer.no_backprop_mode():
        # Update weights using SGD.
        w1 -= eta * w1.grad
        w2 -= eta * w2.grad
        b2 -= eta * b2.grad

        # Clear the gradients for the next iteration.
        w1.cleargrad()
        w2.cleargrad()
        b2.cleargrad()
```


```python
print(w1)
print(w2)
print(b2)
```

    variable([[-6.898038  -6.5185432]
              [ 7.0828695  6.2416263]
              [ 3.471656  -3.3139799]])
    variable([[-11.242552]
              [ 11.863684]])
    variable([5.270227])



```python
F.sigmoid(F.bias(F.matmul(F.sigmoid(F.matmul(x,w1)) ,w2), b2))
```




    variable([[0.00539303],
              [0.9949782 ],
              [0.9927317 ],
              [0.00462809]])



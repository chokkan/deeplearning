

```python
import mxnet as mx
from mxnet import nd, autograd

# Training data for XOR.
x = nd.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = nd.array([[0], [1], [1], [0]])

w1 = nd.random.normal(0, 1, shape=(3, 2))
w2 = nd.random.normal(0, 1, shape=(2, 1))
b2 = nd.random.normal(0, 1, shape=(1, 1))
w1.attach_grad()
w2.attach_grad()
b2.attach_grad()

eta = 0.5
for t in range(1000):
    with autograd.record():
        # y_pred = \sigma(w_2 \cdot \sigma(x \cdot w_1) + b_2)
        y_pred = (nd.dot(nd.dot(x, w1).sigmoid(), w2) + b2).sigmoid()
        ll = y * y_pred + (1 - y) * (1 - y_pred)
        loss = -ll.log().sum()
    loss.backward()
    
    # Update weights using SGD.
    w1 -= eta * w1.grad
    w2 -= eta * w2.grad
    b2 -= eta * b2.grad
```


```python
print(w1)
print(w2)
print(b2)
```

    
    [[  4.5155373   4.041809 ]
     [ -7.8481655   7.4811954]
     [  5.9294176 -10.316905 ]]
    <NDArray 3x2 @cpu(0)>
    
    [[-5.775163 ]
     [-7.2728887]]
    <NDArray 2x1 @cpu(0)>
    
    [[5.775924]]
    <NDArray 1x1 @cpu(0)>



```python
(nd.dot(nd.dot(x, w1).sigmoid(), w2) + b2).sigmoid()
```




    
    [[0.50396055]
     [0.99037385]
     [0.4968157 ]
     [0.00550799]]
    <NDArray 4x1 @cpu(0)>



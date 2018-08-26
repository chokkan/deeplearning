

```python
import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L

dtype=np.float32

# Training data for XOR.
x = chainer.Variable(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype))
y = chainer.Variable(np.array([[0], [1], [1], [0]], dtype=np.int32))

# Define a neural network using high-level modules.
model = chainer.Sequential(
    L.Linear(2, 2, nobias=False),
    F.sigmoid,
    L.Linear(2, 1, nobias=False),
)

# Binary corss-entropy loss after sigmoid function.
loss_fn=F.sigmoid_cross_entropy

# Optimizer based on SGD (change "SGD" to "Adam" to use Adam)
optimizer = chainer.optimizers.SGD(lr=0.5)
optimizer.setup(model)

for t in range(1000):
    y_pred = model(x)                   # Make predictions.
    loss = loss_fn(y_pred, y, normalize=False)  # Compute the loss.
    #print(t, loss.data)
    
    model.cleargrads()          # Zero-clear gradients.
    loss.backward()             # Compute the gradients.
    optimizer.update()          # Update the parameters using the gradients.
```


```python
for param in model.params():
    print(param)
```

    variable b([-2.2869124 -5.1613417])
    variable W([[5.922025  5.878332 ]
                [3.4194984 3.4142656]])
    variable b([-3.3487024])
    variable W([[ 7.281869 -7.459597]])



```python
F.sigmoid(model(x))
```




    variable([[0.06181788],
              [0.93281394],
              [0.93301374],
              [0.08725712]])



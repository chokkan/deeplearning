

```python
import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L
chainer.config.train = True

dtype=np.float32

# Training data for NAND.
x = chainer.Variable(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype))
y = chainer.Variable(np.array([[1], [1], [1], [0]], dtype=np.int32))

# Define a neural network using high-level modules.
model = chainer.Sequential(
    L.Linear(2, 1, nobias=False),   # 2 dims (with bias) -> 1 dim
)

# Binary corss-entropy loss after sigmoid function.
loss_fn = F.sigmoid_cross_entropy

# Optimizer based on SGD (change "SGD" to "Adam" to use Adam)
optimizer = chainer.optimizers.SGD(lr=0.5)
optimizer.setup(model)

for t in range(100):
    y_pred = model(x)                   # Make predictions.
    loss = loss_fn(y_pred, y, normalize=False)   # Compute the loss.
    #print(t, loss.data)
    
    model.cleargrads()          # Zero-clear gradients.
    loss.backward()             # Compute the gradients.
    optimizer.update()          # Update the parameters using the gradients.
```


```python
for para in model.params():
    print(para)
```

    variable b([3.4636898])
    variable W([[-2.134361  -2.1379907]])



```python
F.sigmoid(model(x))
```




    variable([[0.9696368 ],
              [0.79012835],
              [0.7907295 ],
              [0.30817568]])



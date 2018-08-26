

```python
import chainer
import numpy as np
from chainer import Variable, Function
import chainer.functions as F
import chainer.links as L

dtype = np.float32

# Training data for NAND
x = Variable(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype))
y = Variable(np.array([[1], [1], [1], [0]], dtype=np.int32))

# Define a neural network using high-level modules.
model = chainer.Sequential(
    L.Linear(2, 1, nobias=False)            # 2 dims (with bias) -> 1 dim
)
# Binary corss-entropy loss after sigmoid function.
loss_fn=F.sigmoid_cross_entropy

eta = 0.5
for t in range(100):
    y_pred = model(x)                       # Make predictions.
    loss = loss_fn(y_pred, y, normalize=False)
    # print(t, loss.data)
    model.cleargrads()                      # Zero-clear the gradients.
    loss.backward()                         # Compute the gradients.

    with chainer.no_backprop_mode():
        for para in model.params():
            para.data -= eta * para.grad    # Update the parameters using SGD.
```


```python
for para in model.params():
    print(para)
```

    variable b([3.2144895])
    variable W([[-1.9686245 -1.9545643]])



```python
F.sigmoid(model(x))
```




    variable([[0.96137595],
              [0.7790133 ],
              [0.77658325],
              [0.32988632]])



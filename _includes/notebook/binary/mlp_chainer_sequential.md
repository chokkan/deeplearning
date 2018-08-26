

```python
import chainer
import numpy as np
from chainer import Variable, Function
import chainer.functions as F
import chainer.links as L

dtype = np.float32

# Training data for XOR.
x = chainer.Variable(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype))
y = chainer.Variable(np.array([[0], [1], [1], [0]], dtype=np.int32))

# Define a neural network using high-level modules.
init=chainer.initializers.HeNormal()
model = chainer.Sequential(
    L.Linear(2, 2, nobias=False, initialW=init), # 2 dims (with bias) -> 2 dims
    F.sigmoid,                                   # Sigmoid function
    L.Linear(2, 1, nobias=False, initialW=init), # 2 dims (with bias) -> 1 dim
)

# Binary corss-entropy loss after sigmoid function.
loss_fn=F.sigmoid_cross_entropy

eta = 0.5
for t in range(1000):
    y_pred = model(x)                            # Make predictions.
    loss = loss_fn(y_pred, y, normalize=False)
    # print(t, loss.data)
    model.cleargrads()                           # Zero-clear the gradients.
    loss.backward()                              # Compute the gradients.

    with chainer.no_backprop_mode():
        for para in model.params():
            para.data -= eta * para.grad     # Update the parameters using SGD.
```


```python
for para in model.params():
    print(para)
```

    variable W([[-5.0357113  4.694917 ]
                [ 5.884984  -6.006705 ]])
    variable b([-2.5778637 -3.3951974])
    variable W([[7.5983677 7.613726 ]])
    variable b([-3.705806])



```python
F.sigmoid(model(x))
```




    variable([[0.05105227],
              [0.95592314],
              [0.9653981 ],
              [0.04323387]])



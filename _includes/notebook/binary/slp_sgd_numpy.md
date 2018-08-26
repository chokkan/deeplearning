

```python
import numpy as np

def sigmoid(v):
    return 1.0 / (1 + np.exp(-v))

# Training data for NAND.
x = np.array([
    [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
    ])
y = np.array([1, 1, 1, 0])
w = np.array([0.0, 0.0, 0.0])

eta = 0.5
for t in range(100):
    y_pred = sigmoid(np.dot(x, w))
    w -= np.dot((y_pred - y), x)
```


```python
w
```




    array([-5.59504346, -5.59504346,  8.57206068])




```python
sigmoid(np.dot(x, w))
```




    array([0.99981071, 0.95152498, 0.95152498, 0.06798725])



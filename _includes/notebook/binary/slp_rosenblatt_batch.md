

```python
import numpy as np

# Training data for NAND.
x = np.array([
    [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
    ])
y = np.array([1, 1, 1, 0])
w = np.array([0.0, 0.0, 0.0])

eta = 0.5
for t in range(100):
    y_pred = np.heaviside(np.dot(x, w), 0)
    w += np.dot((y - y_pred), x)
```


```python
w
```




    array([-1., -1.,  2.])




```python
np.heaviside(np.dot(x, w), 0)
```




    array([1., 1., 1., 0.])



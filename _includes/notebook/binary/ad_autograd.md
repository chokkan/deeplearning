

```python
import autograd
import autograd.numpy as np

def loss(w, x):
    return -np.log(1.0 / (1 + np.exp(-np.dot(x, w))))

x = np.array([1, 1, 1])
w = np.array([1.0, 1.0, -1.5])

grad_loss = autograd.grad(loss)
print(loss(w, x))
print(grad_loss(w, x))
```

    0.47407698418010663
    [-0.37754067 -0.37754067 -0.37754067]


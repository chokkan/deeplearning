

```python
import numpy as np
from chainer import Variable
import chainer.functions as F

dtype = np.float32

x = np.array([1,1,1], dtype=dtype)
w = Variable(np.array([1.0,1.0,-1.5], dtype=dtype), requires_grad=True)

loss = -F.log(F.sigmoid(np.dot(x,w)))
loss.backward()
print(loss.data)
print(w.grad)
```

    0.47407696
    [-0.37754062 -0.37754062 -0.37754062]


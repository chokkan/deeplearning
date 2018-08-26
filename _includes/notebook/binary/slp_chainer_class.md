

```python
import chainer
import numpy as np
from chainer import Variable, Function
import chainer.functions as F
import chainer.links as L

x = Variable(np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32))
y = Variable(np.array([[1],[1],[1],[0]],dtype=np.int32))

class Linear(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(2,1)
    def __call__(self,x):
        return self.l1(x)

model = Linear()

optimizer = optimizers.SGD(lr=0.5).setup(model)
for t in range(1000):
    y_pred = model(x)
    loss = F.sigmoid_cross_entropy(y_pred,y)
    #print(t,loss.data)
    model.cleargrads()
    loss.backward()
    optimizer.update()
```


```python
F.sigmoid(model(x))
```




    variable([[0.99989665],
              [0.95996976],
              [0.95997   ],
              [0.05611408]])



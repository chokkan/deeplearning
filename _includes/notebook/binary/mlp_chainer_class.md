

```python
import chainer
import numpy as np
from chainer import Variable
from chainer import functions as F
from chainer import links as L
from chainer import optimizers

x = Variable(np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32))
y = Variable(np.array([[0],[1],[1],[0]],dtype=np.int32))

class Linear(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(2,2)
            self.l2 = L.Linear(2,1)
      
    def __call__(self,x):
        h = F.sigmoid(self.l1(x))
        o = self.l2(h)
        return o
    
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




    variable([[0.33073208],
              [0.5335392 ],
              [0.8444923 ],
              [0.26925102]])



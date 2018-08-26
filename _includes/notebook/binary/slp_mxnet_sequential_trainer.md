

```python
import mxnet as mx
from mxnet import nd, autograd, gluon

# Training data for NAND.
x = nd.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = nd.array([[1], [1], [1], [0]])

# Define a neural network using high-level modules.
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1))
net.collect_params().initialize(mx.init.Normal(sigma=1.))
  
# Binary cross-entropy loss agter sigmoid function.
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# Optimizer based on SGD
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for t in range(100):
    with autograd.record():
        # Make predictions.
        y_pred = net(x)
        # Compute the loss.
        loss = loss_fn(y_pred, y)
    # Compute the gradients of the loss.
    loss.backward()
    # Update weights using SGD.
    # the batch_size is set to one to be consistent with the slide.
    trainer.step(batch_size=1)
```


```python
for v in net.collect_params().values():
    print(v, v.data())
```

    Parameter sequential0_dense0_weight (shape=(1, 2), dtype=float32) 
    [[-4.182336  -4.1832795]]
    <NDArray 1x2 @cpu(0)>
    Parameter sequential0_dense0_bias (shape=(1,), dtype=float32) 
    [6.466928]
    <NDArray 1 @cpu(0)>



```python
net(x).sigmoid()
```




    
    [[0.9984484 ]
     [0.90751374]
     [0.9075929 ]
     [0.13025706]]
    <NDArray 4x1 @cpu(0)>





```python
import mxnet as mx
from mxnet import nd, autograd, gluon

# Training data for XOR.
x = nd.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = nd.array([[0], [1], [1], [0]])

# Define a neural network using high-level modules.
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(2))
    net.add(gluon.nn.Activation('sigmoid'))
    net.add(gluon.nn.Dense(1))
net.collect_params().initialize(mx.init.Normal(sigma=1.))

# Binary cross-entropy loss agter sigmoid function.
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# Optimizer based on SGD
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for t in range(1000):
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

    Parameter sequential0_dense0_weight (shape=(2, 3), dtype=float32) 
    [[ 8.537011  -4.4268546  1.9961522]
     [ 8.598794   4.8988695 -1.3610638]]
    <NDArray 2x3 @cpu(0)>
    Parameter sequential0_dense0_bias (shape=(2,), dtype=float32) 
    [ 0.9527137  -0.12632234]
    <NDArray 2 @cpu(0)>
    Parameter sequential0_dense1_weight (shape=(1, 2), dtype=float32) 
    [[-7.1571183  7.3222814]]
    <NDArray 1x2 @cpu(0)>
    Parameter sequential0_dense1_bias (shape=(1,), dtype=float32) 
    [-0.16509056]
    <NDArray 1 @cpu(0)>



```python
net(x).sigmoid()
```




    
    [[0.00362506]
     [0.9962937 ]
     [0.49854442]
     [0.5015438 ]]
    <NDArray 4x1 @cpu(0)>



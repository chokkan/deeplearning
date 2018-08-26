

```python
import torch

dtype = torch.float

# Training data for XOR.
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype)
y = torch.tensor([[0], [1], [1], [0]], dtype=dtype)
                                        
# Define a neural network model.
class ThreeLayerNN(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(ThreeLayerNN, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, d_hidden, bias=True)
        self.linear2 = torch.nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, x):
        return self.linear2(self.linear1(x).sigmoid())

model = ThreeLayerNN(2, 2, 1)

# Binary corss-entropy loss after sigmoid function.
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=False)

# Optimizer based on SGD (change "SGD" to "Adam" to use Adam)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for t in range(1000):
    y_pred = model(x)           # Make predictions.
    loss = loss_fn(y_pred, y)   # Compute the loss.
    #print(t, loss.item())
    
    optimizer.zero_grad()       # Zero-clear gradients.
    loss.backward()             # Compute the gradients.
    optimizer.step()            # Update the parameters using the gradients.
```


```python
model.state_dict()
```




    OrderedDict([('linear1.weight', tensor([[ 6.6212, -6.8110],
                          [ 6.7129, -6.4369]])),
                 ('linear1.bias', tensor([-3.5404,  3.2040])),
                 ('linear2.weight', tensor([[ 11.6606, -11.1694]])),
                 ('linear2.bias', tensor([ 5.2589]))])




```python
model(x).sigmoid()
```




    tensor([[ 0.0058],
            [ 0.9921],
            [ 0.9947],
            [ 0.0049]])



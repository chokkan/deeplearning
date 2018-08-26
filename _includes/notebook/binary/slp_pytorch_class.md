

```python
import torch

dtype = torch.float

# Training data for NAND.
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype)
y = torch.tensor([[1], [1], [1], [0]], dtype=dtype)
                                        
# Define a neural network model.
class SingleLayerNN(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(SingleLayerNN, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, d_out, bias=True)

    def forward(self, x):
        return self.linear1(x)

model = SingleLayerNN(2, 1)

# Binary corss-entropy loss after sigmoid function.
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=False)

# Optimizer based on SGD (change "SGD" to "Adam" to use Adam)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for t in range(100):
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




    OrderedDict([('linear1.weight', tensor([[-4.2693, -4.2689]])),
                 ('linear1.bias', tensor([ 6.5951]))])




```python
model(x).sigmoid()
```




    tensor([[ 0.9986],
            [ 0.9110],
            [ 0.9110],
            [ 0.1253]])



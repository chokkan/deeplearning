

```python
import torch

dtype = torch.float

# Training data for XOR.
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype)
y = torch.tensor([[0], [1], [1], [0]], dtype=dtype)
                                        
# Define a neural network using high-level modules.
model = torch.nn.Sequential(
    torch.nn.Linear(2, 2, bias=True),   # 2 dims (with bias) -> 2 dims
    torch.nn.Sigmoid(),                 # Sigmoid function
    torch.nn.Linear(2, 1, bias=True),   # 2 dims (with bias) -> 1 dim
)

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




    OrderedDict([('0.weight', tensor([[ 7.3702, -7.1611],
                          [-6.6066,  6.9133]])),
                 ('0.bias', tensor([ 3.6234,  3.3088])),
                 ('2.weight', tensor([[-9.1519, -9.2072]])),
                 ('2.bias', tensor([ 13.4994]))])




```python
model(x).sigmoid()
```




    tensor([[ 0.0134],
            [ 0.9826],
            [ 0.9824],
            [ 0.0118]])



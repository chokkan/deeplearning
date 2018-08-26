

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

eta = 0.5
for t in range(1000):
    y_pred = model(x)                   # Make predictions.
    loss = loss_fn(y_pred, y)           # Compute the loss.
    #print(t, loss.item())
    
    model.zero_grad()                   # Zero-clear the gradients.
    loss.backward()                     # Compute the gradients.
        
    with torch.no_grad():
        for param in model.parameters():
            param -= eta * param.grad   # Update the parameters using SGD.
```


```python
model.state_dict()
```




    OrderedDict([('0.weight', tensor([[ 7.0281,  7.0367],
                          [ 5.1955,  5.1971]])),
                 ('0.bias', tensor([-3.1767, -7.9526])),
                 ('2.weight', tensor([[ 11.4025, -12.1782]])),
                 ('2.bias', tensor([-5.2898]))])




```python
model(x).sigmoid()
```




    tensor([[ 0.0079],
            [ 0.9942],
            [ 0.9942],
            [ 0.0061]])





```python
import torch

dtype = torch.float

# Training data for NAND.
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype)
y = torch.tensor([[1], [1], [1], [0]], dtype=dtype)
                                        
# Define a neural network using high-level modules.
model = torch.nn.Sequential(
    torch.nn.Linear(2, 1, bias=True),   # 2 dims (with bias) -> 1 dim
)

# Binary corss-entropy loss after sigmoid function.
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=False)

eta = 0.5
for t in range(100):
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




    OrderedDict([('0.weight', tensor([[-4.3067, -4.3060]])),
                 ('0.bias', tensor([ 6.6506]))])




```python
model(x).sigmoid()
```




    tensor([[ 0.9987],
            [ 0.9125],
            [ 0.9124],
            [ 0.1232]])



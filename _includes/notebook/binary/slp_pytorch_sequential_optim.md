

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




    OrderedDict([('0.weight', tensor([[-4.2726, -4.2721]])),
                 ('0.bias', tensor([ 6.6000]))])




```python
model(x).sigmoid()
```




    tensor([[ 0.9986],
            [ 0.9112],
            [ 0.9111],
            [ 0.1251]])



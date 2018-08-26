

```python
import torch

dtype = torch.float

# Training data for NAND.
x = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=dtype)
y = torch.tensor([[1], [1], [1], [0]], dtype=dtype)
w = torch.randn(3, 1, dtype=dtype, requires_grad=True)

eta = 0.5
for t in range(100):
    # y_pred = \sigma(x \cdot w)
    y_pred = x.mm(w).sigmoid()
    ll = y * y_pred + (1 - y) * (1 - y_pred)
    loss = -ll.log().sum()      # The loss value.
    #print(t, loss.item())
    loss.backward()             # Compute the gradients of the loss.

    with torch.no_grad():
        w -= eta * w.grad       # Update weights using SGD.        
        w.grad.zero_()          # Clear the gradients for the next iteration.
```


```python
w
```




    tensor([[-4.2327],
            [-4.2320],
            [ 6.5405]])




```python
x.mm(w).sigmoid()
```




    tensor([[ 0.9986],
            [ 0.9096],
            [ 0.9095],
            [ 0.1274]])



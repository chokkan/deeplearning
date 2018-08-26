

```python
import torch

dtype = torch.float

# Training data for XOR.
x = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=dtype)
y = torch.tensor([[0], [1], [1], [0]], dtype=dtype)
w1 = torch.randn(3, 2, dtype=dtype, requires_grad=True)
w2 = torch.randn(2, 1, dtype=dtype, requires_grad=True)
b2 = torch.randn(1, 1, dtype=dtype, requires_grad=True)

eta = 0.5
for t in range(1000):
    # y_pred = \sigma(w_2 \cdot \sigma(x \cdot w_1) + b_2)
    y_pred = x.mm(w1).sigmoid().mm(w2).add(b2).sigmoid()
    ll = y * y_pred + (1 - y) * (1 - y_pred)
    loss = -ll.log().sum()
    #print(t, loss.item())
    loss.backward()
    
    with torch.no_grad():
        # Update weights using SGD.
        w1 -= eta * w1.grad
        w2 -= eta * w2.grad
        b2 -= eta * b2.grad
        
        # Clear the gradients for the next iteration.
        w1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()
```


```python
print(w1)
print(w2)
print(b2)
```

    tensor([[ 5.1994,  7.0126],
            [ 5.2027,  7.0301],
            [-7.9606, -3.1728]])
    tensor([[-12.1454],
            [ 11.3713]])
    tensor([[-5.2753]])



```python
x.mm(w1).sigmoid().mm(w2).add(b2).sigmoid()
```




    tensor([[ 0.0080],
            [ 0.9942],
            [ 0.9941],
            [ 0.0062]])



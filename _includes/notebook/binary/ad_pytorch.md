

```python
import torch

dtype = torch.float

x = torch.tensor([1, 1, 1], dtype=dtype)
w = torch.tensor([1.0, 1.0, -1.5], dtype=dtype, requires_grad=True)

loss = -torch.dot(x, w).sigmoid().log()
loss.backward()
print(loss.item())
print(w.grad)
```

    0.4740769565105438
    tensor([-0.3775, -0.3775, -0.3775])


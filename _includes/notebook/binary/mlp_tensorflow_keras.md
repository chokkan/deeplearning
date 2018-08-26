

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras import optimizers

# Training data for XOR.
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

# Define a neural network using high-level modules.
model = Sequential([
    Flatten(),
    Dense(2, activation='sigmoid'),    # 2 dims (with bias) -> 2 dims
    Dense(1, activation='sigmoid')     # 2 dims (with bias) -> 2 dims
])

model.compile(
    optimizer=optimizers.SGD(lr=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

model.fit(x_data, y_data, epochs=1000, verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x7fd00b4d00f0>




```python
model.get_weights()
```




    [array([[2.0502675, 5.6302657],
            [1.0301563, 4.942775 ]], dtype=float32),
     array([-1.615101 , -1.5490855], dtype=float32),
     array([[-4.0722337],
            [ 5.53848  ]], dtype=float32),
     array([-2.3617785], dtype=float32)]




```python
model.predict(x_data)
```




    array([[0.11236145],
           [0.8234226 ],
           [0.6484979 ],
           [0.4670412 ]], dtype=float32)



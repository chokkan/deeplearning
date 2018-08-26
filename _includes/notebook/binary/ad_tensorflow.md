

```python
import tensorflow as tf

x = tf.constant([1., 1., 1.])
w = tf.Variable([1.0, 1.0, -1.5])

loss = -tf.log(tf.sigmoid(tf.tensordot(x, w, axes=1)))
grad = tf.gradients(loss, w)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_value = sess.run(loss)
    grad_value = sess.run(grad)
    print(loss_value)
    print(grad_value)
```

    0.47407696
    [array([-0.37754062, -0.37754062, -0.37754062], dtype=float32)]


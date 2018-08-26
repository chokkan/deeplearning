

```python
import tensorflow as tf

# Training data for NAND
x_data = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
y_data = [[1], [1], [1], [0]]

x = tf.placeholder(tf.float32, [4, 3])
y = tf.placeholder(tf.float32, [4, 1])
w = tf.Variable(tf.random_normal([3,1]))

# y_pred = \sigma(x \cdot w)
y_pred = tf.sigmoid(tf.matmul(x, w))
ll = y * y_pred + (1 - y) * (1 - y_pred)
loss = -tf.reduce_sum(tf.log(ll))
grad = tf.gradients(loss, w)

eta = 0.5
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for t in range(100):
        grads = sess.run(grad, feed_dict={x: x_data, y: y_data})
        sess.run(w.assign_sub(eta * grads[0]))
    print(sess.run(w))
    print(sess.run(y_pred, feed_dict={x: x_data, y: y_data}))
```

    [[-4.203982 ]
     [-4.2044005]
     [ 6.4987044]]
    [[0.9984969 ]
     [0.90840423]
     [0.90843904]
     [0.12901704]]


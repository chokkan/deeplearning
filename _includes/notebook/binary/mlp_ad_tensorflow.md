

```python
import tensorflow as tf

# Training data for XOR.
x_data = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
y_data = [[0], [1], [1], [0]]

x = tf.placeholder(tf.float32, [4, 3])
y = tf.placeholder(tf.float32, [4, 1])
w1 = tf.Variable(tf.random_normal([3, 2]))
w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.random_normal([1, 1]))

y_pred = tf.sigmoid(tf.add(tf.matmul(tf.sigmoid(tf.matmul(x, w1)), w2), b2))
ll = y * y_pred + (1 - y) * (1 - y_pred)
log = tf.log(ll)
loss = -tf.reduce_sum(log)
grad = tf.gradients(loss, [w1, w2, b2])

eta = 0.5
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(1000):
        w1_grad, w2_grad, b2_grad = sess.run(grad, feed_dict={x: x_data, y: y_data})
        sess.run(tf.assign_sub(w1, eta * w1_grad))
        sess.run(tf.assign_sub(w2, eta * w2_grad))
        sess.run(tf.assign_sub(b2, eta * b2_grad))
        
    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(b2))
    print(sess.run(y_pred, feed_dict={x: x_data, y: y_data}))
```

    [[ 8.545845   8.527924 ]
     [ 4.8393583 -4.2591324]
     [-1.4335978  2.7701557]]
    [[ 7.452827 ]
     [-7.1250057]]
    [[-0.32773858]]
    [[0.00369266]
     [0.9962198 ]
     [0.49852544]
     [0.5015694 ]]


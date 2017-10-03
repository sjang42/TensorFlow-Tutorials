import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable('W1', [3, 3, 1, 32], tf.float32,
                     tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(
    L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable('W2', [3, 3, 32, 64], tf.float32,
                     tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(
    L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable('W3', [3, 3, 64, 128], tf.float32,
                     tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(
    L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable('W4', [4 * 4 * 128, 256], tf.float32,
                     tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable('b4', [256], tf.float32,
                     tf.contrib.layers.xavier_initializer())
L4 = tf.reshape(L3, [-1, 4 * 4 * 128])
L4 = tf.matmul(L4, W4) + b4
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', [256, 10], tf.float32,
                     tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable('b5', [10], tf.float32,
                     tf.contrib.layers.xavier_initializer())

logits = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run(
            [optimizer, cost],
            feed_dict={X: batch_xs,
                       Y: batch_ys,
                       keep_prob: 0.7})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost:', '{:3f}'.format(
        total_cost / total_batch))

print('최적화 완료!')

is_correct(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('accuracy :',
      sess.run(
          accuracy,
          feed_dict={
              X: mnist.test.images.reshape(-1, 28, 28, 1),
              Y: mnist.test.labels,
              keep_prob: keep_prob
          }))

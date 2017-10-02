import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28 * 28])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.get_variable(
        'W1', [28 * 28, 512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(
        'b1', [512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob)

with tf.name_scope('layer2'):
    W2 = tf.get_variable(
        'W2', [512, 512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(
        'b2', [512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob)

with tf.name_scope('layer3'):
    W3 = tf.get_variable(
        'W3', [512, 512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(
        'b3', [512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob)


with tf.name_scope('layer4'):
    W4 = tf.get_variable(
        'W4', [512, 512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable(
        'b4', [512],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob)

with tf.name_scope('output'):
    W5 = tf.get_variable(
        'W5', [512, 10],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable(
        'b5', [10],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch = 15
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for steps in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}

        _, cost_val = sess.run([train_op, cost], feed_dict=feed_dict)
        total_cost += cost_val

    print('Epoch:', '%04d' % (steps + 1), 'Avg. cost =', 
        total_cost / total_batch)

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('accuracy : ',
      sess.run(
          accuracy, feed_dict={X: mnist.test.images,
                               Y: mnist.test.labels, keep_prob: 1.}))

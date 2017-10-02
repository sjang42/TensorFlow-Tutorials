import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(0, trainable=False, name='gloabal_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10]), name='W1')
W2 = tf.Variable(tf.random_uniform([10, 20]), name='W2')
W3 = tf.Variable(tf.random_uniform([20, 3]), name='W3')

b1 = tf.Variable(tf.random_uniform([10]), name='bias1')
b2 = tf.Variable(tf.random_uniform([20]), name='bias2')
b3 = tf.Variable(tf.random_uniform([3]), name='bias3')

L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
logits = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

feed_dict = {X: x_data, Y: y_data}
for step in range(2):
    sess.run(train_op, feed_dict=feed_dict)

    print("Step: %d" % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict=feed_dict))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

prediction = tf.argmax(logits, 1)
target = tf.argmax(Y, 1)

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print("accuracy : %.2f" % sess.run(accuracy, feed_dict=feed_dict))

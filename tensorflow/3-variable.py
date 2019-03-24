# use variable

import tensorflow as tf

state = tf.Variable(1, name='counter')

one = tf.constant(1)

new_value = tf.add(state, one)

update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
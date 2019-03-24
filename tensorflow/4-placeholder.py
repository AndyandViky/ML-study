# use placehodel 申领内存，可替换其中的值

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1: [2.], input2: [2.1]})
    print(result)


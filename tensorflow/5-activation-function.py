# use activation function

# https://blog.csdn.net/u011630575/article/details/78063641

# 可视化 tensorboard --logdir='logs/'

import tensorflow as tf
import numpy as np


def add_layer(intputs, in_size, out_size, n_layer, activation_function=None):  # add new laryer
    layer_name = "layer%s" % n_layer
    with tf.name_scope(layer_name):

        with tf.name_scope("Weight"):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name+'/Weights', Weight)

        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]), name="b") + 0.1  # 推荐大于0
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(intputs, Weight) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)

        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 创建等差数列
noise = np.random.normal(0, 0.05, x_data.shape)  # 模拟噪点
y_data = np.square(x_data) - 0.5 + noise  # 创建结果数据集

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="input_x")  # 创建x_data占位符
    ys = tf.placeholder(tf.float32, [None, 1], name="input_y")  # 创建y_data占位符

layer1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu)  # 创建输入层
prediction = add_layer(layer1, 10, 1, 2, activation_function=None)  # 创建输出层

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # 计算损失
    tf.summary.scalar('loss', loss)

with tf.name_scope("train"):
    train_setps = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 最小化loss

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)  # 将整个可视化文件写入本地

    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(train_setps, feed_dict={xs: x_data, ys: y_data})
        if step % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, step)
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

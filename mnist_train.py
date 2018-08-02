
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载mnist_inference.py中定义的常量和前向传播的函数。
import mnist_inference
import numpy as np

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECARY = 0.99  # 滑动平均的衰减率
# 模型的保存路径以及名称



def train(mnist):
    # 定义输入输出placeholder.
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,  # 第一维表示一个batch中样例的个数
                        mnist_inference.IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸。
                        mnist_inference.IMAGE_SIZE,
                        mnist_inference.NUM_CHANNELS],  # 第四维度表示图片的深度
                       name='x-input'
                       )
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    # 直接使用mnist_inferience.py中的前向传播过程。
    y = mnist_inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数，滑动平均率，学习率，以及训练过程。
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECARY, global_step)
    variable_average_op = variable_average.apply(
        tf.trainable_variables()
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # y是正确的数字只有一个，y_是输出的数字有十个选出最大的一个
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 训练过程中不再验证，测试与验证放在另一个程序中
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            # 每1000轮保存一次模型。
            if i % 10 == 0:
                # 输出当前训练情况，这里只输出了模型在当前训练batch上的损失函数，通过这个来近似了解当前训练情况。
                # 在验证数据上的正确信息会有一个单独的程序完成。
                print("After %d training step(s),loss on training " "batch is %g." % (step, loss_value))
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个模型文件名最后都加上训练的轮数，比如

                # saver这个类里面带的函数最后由有参数可以自动加上步数，global_step


def main(argv=None):
    mnist = input_data.read_data_sets("MINST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
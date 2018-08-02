import tensorflow as tf
#前两个参数表示是过滤器的尺寸，第三个是当前维度的深度，第四个是当前层的深度，第四个是过滤器的深度
filter_weight=tf.get_variable(
    'weights',[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1)
)
#当前层矩阵上不同位置上的偏置项是共享的，所以一个深度共用一个偏置项，上面过滤器有16个深度所以要有16
#个偏置项
biases=tf.get_variable(
    'biases',[16],initializer=tf.constant_initializer(0.1)
)
#tf.nn.conv2d第一个参数是当前层的节点矩阵（是一个四维矩阵），后面三个维度对应一个节点矩阵，
#第一维对应一个输入batch，第二个参数提供权重，第三个参数提供不同维度的步长，最后一个选择填充样式
#‘SAME’表示全零填充，‘VAILD’表示不填充
conv=tf.nn.conv2d(
    input,filter_weight,strides=[1,1,1,1],padding='SAME')

#tf.nn.bias_add()提供了一个方便的函数给每个节点加上偏置项，注意这里不能使用加法，因为矩阵同一层的
#不同位置都要加上相同偏置项
bias=tf.nn.bias_add(conv,biases)
#将计算结果通过RELU函数激活完成去线性化
actived_conv=tf.nn.relu(bias)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MINST_data",one_hot=True)
#训练批次
batch_size =100
#计算有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个放数据的地方 placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])

#创建一个神经网络 首要的就是权值和偏置
with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        w = tf.Variable(tf.truncated_normal([784,10],stddev = 0.1))
    with tf.name_scope('baise'):
        b = tf.Variable(tf.zeros([10]) + 0.1) #默认10列
    with tf.name_scope('W_x_plus_b'):
        W_x_plus_b = tf.matmul(x,w) + b
    with tf.name_scope('prediction'):
       prediction = tf.nn.softmax(W_x_plus_b) #激励函数

#二次代价函数  即损失函数,是我们自己定义的
#loss =tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=(tf.matmul(x,w) + b))

#用Adam下降法降低loss
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#为了显示准确率，定义一个bool类型变量
with tf.name_scope('Toacc'):
    with tf.name_scope('true'):
        T_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#arg_max是返回tensor中最大值的位置
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(T_prediction,tf.float32))#cast将bool类型转换成浮点型

#放入session中运行
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs',sess.graph)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#得到下一个batch数量的训练数据
            sess.run(train_step,feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ", Test accuracy" + str(acc) + " ,loss:" +  str(loss))

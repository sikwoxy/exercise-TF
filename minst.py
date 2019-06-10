import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MINST_data",one_hot=True)
#训练批次
batch_size =20
#计算有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个放数据的地方 placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#创建一个神经网络 首要的就是权值和偏置
w1 = tf.Variable(tf.truncated_normal([784,300],stddev = 0.1))
b1 = tf.Variable(tf.zeros([300]) + 0.1) #默认10列
#prediction = tf.nn.softmax(tf.matmul(x,w) + b) #激励函数
l1 = tf.nn.tanh(tf.matmul(x,w1) + b1)
l1_drop = tf.nn.dropout(l1,keep_prob) #修建其中多余的节点

#创建隐藏层
w2 = tf.Variable(tf.truncated_normal([300,100],stddev = 0.1))
b2 = tf.Variable(tf.zeros([100])+0.1)
l2 = tf.nn.tanh(tf.matmul(l1_drop,w2) + b2)
l2_drop = tf.nn.dropout(l2,keep_prob)

#输出层
w3 = tf.Variable(tf.truncated_normal([100,10],stddev = 0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(l2_drop, w3) + b3)

#二次代价函数  即损失函数,是我们自己定义的
#loss =tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

#用梯度下降法降低loss
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#为了显示准确率，定义一个bool类型变量
T_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#arg_max是返回tensor中最大值的位置
accuracy = tf.reduce_mean(tf.cast(T_prediction,tf.float32))#cast将bool类型转换成浮点型

#放入session中运行
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#得到下一个batch数量的训练数据
            sess.run(train_step,feed_dict={x: batch_xs, y: batch_ys,keep_prob:1.0})

        acc = sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})
        print("Iter" + str(epoch) + ", Test accuracy" + str(acc))

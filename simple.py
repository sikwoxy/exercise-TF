import tensorflow as tf
import numpy as np

x_data =np.random.rand(100)
y_data =x_data * 5 + 1.1

# 定义模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

#loss函数
loss = tf.reduce_mean(tf.square(y - y_data))
#定义一个梯度下降器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#用上述的梯度下降法不断减少loss
train = optimizer.minimize(loss)
#初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(300):
        sess.run(train)
        if step%30 == 0:
            print(step,sess.run([k,b]))

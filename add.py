import tensorflow as tf

var1 =tf.constant([1,2])
var2 =tf.constant([5,4])

add =tf.add(var1,var2)

with tf.Session() as sess:
    sess.run(add);
    print(sess.run(add))

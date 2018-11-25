# 学習させる(for)
def execution(num, sess, x, p, t, loss, train_step, correct_prediction, accuracy, train_x, train_t, y):
    i = 0
    for _ in range(num):
        i += 1
        sess.run(train_step, feed_dict={x:train_x, t:train_t})
        if i % 100 == 0:
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:train_x, t:train_t})
            result = sess.run(y, feed_dict={x: train_x})
            print ('Step: %d, Loss: %f, Accuracy: %f,'% (i, loss_val, acc_val))
            print(str(result[0]))

def learning_algorithm(tf, data_len):
    num_units1 = 200
    num_units2 = 200
    num_units3 = 200

    x = tf.placeholder(tf.float32, [None, data_len])

    w1 = tf.Variable(tf.truncated_normal([data_len, num_units1]))
    b1 = tf.Variable(tf.zeros([num_units1]))
    hidden1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
    b2 = tf.Variable(tf.zeros([num_units2]))
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2) + b2)

    w3 = tf.Variable(tf.truncated_normal([num_units2, num_units3]))
    b3 = tf.Variable(tf.zeros([num_units3]))
    hidden3 = tf.nn.tanh(tf.matmul(hidden2, w3) + b3)

    w0 = tf.Variable(tf.zeros([num_units2, 1]))
    b0 = tf.Variable(tf.zeros([1]))
    #y = tf.matmul(hidden2, w0) + b0
    y = tf.matmul(hidden3, w0) + b0
    p = tf.nn.sigmoid(y)

    t = tf.placeholder(tf.float32, [None, 1])
    loss = -tf.reduce_sum(t*tf.log(tf.clip_by_value(p,1e-10,1.0)) + (1-t)*tf.log(1-tf.clip_by_value(p,1e-10,1.0)))
    # lossを小さくして行く方法
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # equalは、一致しているかどうか。signは符号を返す
    correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
    # 与えたリストの平均値を求める。castは型変換。correct_prediction(true or false)を0 or 1に
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, p, t, loss, train_step, correct_prediction, accuracy, y
#return loss, accuracy, train_step, p


# アルゴリズム

# 引数
# tf, data_len
# return で必要そうなの
# loss, accuracy train_x, train_t, p
# 隠れ層

def double_layer(tf, data_len):
    num_units1 = 16
    num_units2 = 16
    
    x = tf.placeholder(tf.float32, [None, data_len])
    
    w1 = tf.Variable(tf.truncated_normal([data_len, num_units1]))
    b1 = tf.Variable(tf.zeros([num_units1]))
    hidden1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
    
    w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
    b2 = tf.Variable(tf.zeros([num_units2]))
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2) + b2)
    
    w0 = tf.Variable(tf.zeros([num_units2, 1]))
    b0 = tf.Variable(tf.zeros([1]))
    p = tf.nn.sigmoid(tf.matmul(hidden2, w0) + b0)
    
    t = tf.placeholder(tf.float32, [None, 1])
    loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))
    # lossを小さくして行く方法
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # equalは、一致しているかどうか。signは符号を返す
    correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
    # 与えたリストの平均値を求める。castは型変換。correct_prediction(true or false)を0 or 1に
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return x, p, t, loss, train_step, correct_prediction, accuracy
    #return loss, accuracy, train_step, p
    

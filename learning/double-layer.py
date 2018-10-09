import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame, Series
import os
import wave


np.random.seed(20160615)
tf.set_random_seed(20160615)


#def generate_datablock(data_len, columns, n, mu, var, t):
#    data = multivariate_normal(mu, np.eye(data_len)*var, n)
#    df = DataFrame(data, columns=columns)
#    df['t'] = t
#    return df

def wave_fft(file_name):
    wave_file = wave.open(file_name, "r")
    buf = wave_file.readframes(wave_file.getnframes())
    wave_file.close()
    x = np.frombuffer(buf, dtype="int16") / 32768.0
    X = np.fft.fft(x)
    amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
    return amplitudeSpectrum

finger_wav_files = os.listdir("./sounds/finger")
not_finger_wav_files = os.listdir("./sounds/not-finger")

data_len = 2048
#columns = ["x" + str(i + 1) for i in range(data_len)]

#理想 TODO
train_x = np.empty((0, data_len), int)
train_t = np.array([])

for file_name in finger_wav_files:
    file_name = "./sounds/finger/" + file_name
    amplitudeSpectrum = wave_fft(file_name)
    train_x = np.append(train_x, np.array([amplitudeSpectrum]), axis=0)
    train_t = np.append(train_t, 1)

for file_name in not_finger_wav_files:
    file_name = "./sounds/not-finger/" + file_name
    amplitudeSpectrum = wave_fft(file_name)
    train_x = np.append(train_x, np.array([amplitudeSpectrum]), axis=0)
    train_t = np.append(train_t, 0)

train_t = train_t.reshape([len(train_t), 1])
print(train_x.shape)
print(train_t.shape)
# TODO 
    
#df_finger = generate_tff_datablock(data_len, columns, finger_wav_files, 1)
#df_not_finger = generate_tff_datablock(data_len, columns, not_finger_wav_files, 0)
#df = pd.concat([df_finger, df_not_finger], ignore_index=True)

#df0 = generate_datablock(data_len, columns, 30, [-7,-7], 18, 1)
#df1 = generate_datablock(data_len, columns, 30, [-7,7], 18, 0)
#df2 = generate_datablock(data_len, columns, 30, [7,-7], 18, 0)
#df3 = generate_datablock(data_len, columns, 30, [7,7], 18, 1)
#df = pd.concat([df0, df1, df2, df3], ignore_index=True)

#train_set = df.reindex(permutation(df.index)).reset_index(drop=True)
#
#train_x = train_set[columns].as_matrix()
#train_t = train_set['t'].as_matrix().reshape([len(train_set), 1])

# 隠れ層
num_units1 = 4
num_units2 = 4

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
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())

i = 0
for _ in range(2000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    if i % 100 == 0:
        loss_val, acc_val = sess.run(
            [loss, accuracy], feed_dict={x:train_x, t:train_t})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))

# --- テスト ---
result = sess.run(p, feed_dict={x: [train_x[0]]})
print(train_t[0])
print(result)

# --- graph ---
#train_set1 = train_set[train_set['t']==1]
#train_set2 = train_set[train_set['t']==0]
#
#fig = plt.figure(figsize=(6,6))
#subplot = fig.add_subplot(1,1,1)
#subplot.set_ylim([-15,15])
#subplot.set_xlim([-15,15])
#subplot.scatter(train_set1.x1, train_set1.x2, marker='x')
#subplot.scatter(train_set2.x1, train_set2.x2, marker='o')
#
#locations = []
#for x2 in np.linspace(-15,15,100):
#    for x1 in np.linspace(-15,15,100):
#        locations.append((x1,x2))
#p_vals = sess.run(p, feed_dict={x:locations})
#p_vals = p_vals.reshape((100,100))
#subplot.imshow(p_vals, origin='lower', extent=(-15,15,-15,15),
#               cmap=plt.cm.gray_r, alpha=0.5)

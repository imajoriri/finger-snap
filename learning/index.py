import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame, Series
import os
import wave
import random

import learning_algorithm
import learning
import sound_data

np.random.seed(20160615)
tf.set_random_seed(20160615)

project_dir = "/Users/imajo/Desktop/dev/google-assistant-mac/finger-snap/"
finger_wav_files = os.listdir(project_dir + "./sounds/finger")
not_finger_wav_files = os.listdir(project_dir + "./sounds/not-finger")

data_len = 2048

train_x = np.empty((0, data_len), int)
train_t = np.array([])
train_x, train_t = sound_data.add_train_data(project_dir + "./sounds/finger/", finger_wav_files, train_x, train_t, 1)
train_x, train_t = sound_data.add_train_data(project_dir + "./sounds/not-finger/", not_finger_wav_files, train_x, train_t, 0)

train_t = train_t.reshape([len(train_t), 1])

# アルゴリズム作成
x, p, t, loss, train_step, correct_prediction, accuracy, y = learning_algorithm.double_layer(tf, data_len)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# TODO 保存を読み込む場合
#saver.restore(sess, project_dir + "./model_data/model.ckpt")

# 学習の実行
learning_num = 10000
learning.execution(learning_num, sess, x, p, t, loss, train_step, correct_prediction, accuracy, train_x, train_t, y)

print("正解データ数は:" + str(len(finger_wav_files)))
print("不正解データ数は:" + str(len(not_finger_wav_files)))
print("学習回数は: " + str(learning_num))

# TODO 保存する場合
saver.save(sess, project_dir + "./model_data/model.ckpt")

# --- テスト ---
#result = sess.run(p, feed_dict={x: [train_x[10]]})
print(type(train_x[0]))
result = sess.run(p, feed_dict={x: train_x})
#for i in range(len(result)):
#    print(str(result[i]) + ":" + str(train_t[i]))

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

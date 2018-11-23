"""
学習を実行するファイル
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame, Series
import os
import wave
import random

project_dir = os.getcwd() + "/"
# 自作モジュールのimport
import sys
sys.path.append(project_dir + "my_modules/")
import learning
import sound_data

np.random.seed(20160615)
tf.set_random_seed(20160615)

# 正解データのファイル一覧取得
finger_wav_files = os.listdir(project_dir + "./sounds/finger")
# 不正解データのファイル一覧取得
not_finger_wav_files = os.listdir(project_dir + "./sounds/not-finger")

# 単発音のデータの長さ
data_len = 2048

# トレーニングデータの取得
train_x = np.empty((0, data_len), int)
train_t = np.array([])
train_x, train_t = sound_data.add_train_data(project_dir + "./sounds/finger/", finger_wav_files, train_x, train_t, 1)
train_x, train_t = sound_data.add_train_data(project_dir + "./sounds/not-finger/", not_finger_wav_files, train_x, train_t, 0)

train_t = train_t.reshape([len(train_t), 1])

# tensorflowで必要な変数を取得
x, p, t, loss, train_step, correct_prediction, accuracy, y = learning.learning_algorithm(tf, data_len)

# tfのセッション初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# TODO 保存を読み込む場合
saver = tf.train.Saver()
#saver.restore(sess, project_dir + "./model_data/model.ckpt")

# 学習の実行数
learning_num = 10000

# 学習の実行
learning.execution(learning_num, sess, x, p, t, loss, train_step, correct_prediction, accuracy, train_x, train_t, y)

print("正解データ数は:" + str(len(finger_wav_files)))
print("不正解データ数は:" + str(len(not_finger_wav_files)))
print("学習回数は: " + str(learning_num))

# TODO 保存する場合
saver.save(sess, project_dir + "./model_data/model.ckpt")

# --- テスト ---
#result = sess.run(p, feed_dict={x: [train_x[10]]})
#print(type(train_x[0]))
#result = sess.run(p, feed_dict={x: train_x})
#for i in range(len(result)):
#    print(str(result[i]) + ":" + str(train_t[i]))


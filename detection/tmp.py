import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
#import pandas as pd
#from pandas import DataFrame, Series
import os
import wave
import random

import sys
project_dir = "/Users/imajo/Desktop/dev/google-assistant-mac/finger-snap/"
sys.path.append(project_dir + "learning/")
import learning_algorithm
import learning
import sound_data

np.random.seed(20160615)
tf.set_random_seed(20160615)

project_dir = "/Users/imajo/Desktop/dev/google-assistant-mac/finger-snap/"
finger_wav_files = os.listdir(project_dir + "./sounds/finger")
not_finger_wav_files = os.listdir(project_dir + "./sounds/not-finger")

data_len = 2048

#train_x = np.empty((0, data_len), int)
#train_t = np.array([])
#train_x, train_t = sound_data.add_train_data(project_dir + "./sounds/finger/", finger_wav_files, train_x, train_t, 1)
#train_x, train_t = sound_data.add_train_data(project_dir + "./sounds/not-finger/", not_finger_wav_files, train_x, train_t, 0)

#train_t = train_t.reshape([len(train_t), 1])

# アルゴリズム作成
x, p, t, loss, train_step, correct_prediction, accuracy = learning_algorithm.double_layer(tf, data_len)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# TODO 保存を読み込む場合
saver.restore(sess, project_dir + "./model_data/model.ckpt")



file_name = project_dir + "sounds/finger/20181007214123.wav"
wave_file = wave.open(file_name, "r")
buf = wave_file.readframes(wave_file.getnframes())
wave_file.close()
data = np.frombuffer(buf, dtype="int16") / 32768.0
X = np.fft.fft(data)
amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]

train_x = np.empty((0, data_len), int)
train_x = np.insert(train_x, 0, np.array([amplitudeSpectrum]), axis=0)
#result = sess.run(p, feed_dict={x: train_x })

result = sess.run(p, feed_dict={x: train_x})
print('----')
print(result)



# -*- coding: utf-8 -*-

"""
学習を実行するファイル
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os

project_dir = os.getcwd() + "/"
# 自作モジュールのimport
import sys
sys.path.append(project_dir + "my_modules/")
import learning
import sound_data
import const

np.random.seed(20160615)
tf.set_random_seed(20160615)

def set_tensorflow():
    # トレーニングデータを入れる変数の取得
    train_x = np.empty((0, const.FOR_TENSORFLOW.DATA_LEN), int)
    train_t = np.array([])
    
    # RATEによって取得するデータを変更
    if const.FOR_PYAUDIO.RATE == 44100:
    
        finger_wav_files = os.listdir(project_dir + "./sounds/finger-44100")
        not_finger_wav_files = os.listdir(project_dir + "./sounds/not-finger-44100")
    
        train_x, train_t = sound_data.add_train_data(project_dir 
                + "./sounds/finger-44100/", finger_wav_files, train_x, train_t, 1)
        train_x, train_t = sound_data.add_train_data(project_dir 
                + "./sounds/not-finger-44100/", not_finger_wav_files, train_x, train_t, 0)
    
    elif const.FOR_PYAUDIO.RATE == 16000:
    
        finger_wav_files = os.listdir(project_dir + "./sounds/finger-16000")
        not_finger_wav_files = os.listdir(project_dir + "./sounds/not-finger-16000")
    
        train_x, train_t = sound_data.add_train_data(project_dir 
                + "./sounds/finger-16000/", finger_wav_files, train_x, train_t, 1)
        train_x, train_t = sound_data.add_train_data(project_dir
                + "./sounds/not-finger-16000/", not_finger_wav_files, train_x, train_t, 0)
    
    train_t = train_t.reshape([len(train_t), 1])
    
    ## tfのセッション初期化
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    return sess, train_x, train_t, finger_wav_files, not_finger_wav_files

"""
学習データの読み込み
"""
def get_tensorflow_data(sess):
    saver = tf.train.Saver()
    if const.FOR_PYAUDIO.RATE == 44100:
        #saver.restore(sess, project_dir + "./model_data_44100/model.ckpt")
        print("41000")
    elif const.FOR_PYAUDIO.RATE == 16000:
        #saver.restore(sess, project_dir + "./model_data_16000/model.ckpt")
        print("16000")

    return saver

"""
学習データの保存
"""
def save_tensorflow_data(saver, sess):
    if const.FOR_PYAUDIO.RATE == 44100:
        saver.save(sess, project_dir + "./model_data_44100/model.ckpt")
    elif const.FOR_PYAUDIO.RATE == 16000:
        saver.save(sess, project_dir + "./model_data_16000/model.ckpt")

    return saver

def main():
    x, p, t, loss, train_step, correct_prediction, accuracy, y = learning.learning_algorithm(tf, const.FOR_TENSORFLOW.DATA_LEN)
    sess, train_x, train_t, finger_wav_files, not_finger_wav_files = set_tensorflow()
    saver = get_tensorflow_data(sess)

    # 学習の実行数
    learning_num = 4000
    
    # 学習の実行
    learning.execution(learning_num, sess, x, p, t, loss, train_step, correct_prediction, accuracy, train_x, train_t, y)
    
    #print("正解データ数は:" + str(len(finger_wav_files)))
    #print("不正解データ数は:" + str(len(not_finger_wav_files)))
    #print("学習回数は: " + str(learning_num))

    save_tensorflow_data(saver, sess)
    
    # --- テスト ---
    #result = sess.run(p, feed_dict={x: [train_x[10]]})
    #print(type(train_x[0]))
    #result = sess.run(p, feed_dict={x: train_x})
    #for i in range(len(result)):
    #    print(str(result[i]) + ":" + str(train_t[i]))
    
if __name__ == "__main__":
    main()

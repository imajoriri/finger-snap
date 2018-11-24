"""
実際に指パッチンを検出するファイル
以下で実行
python detection.py (実行する秒数)
"""

import pyaudio
import sys
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tensorflow as tf
import os

project_dir = os.getcwd() + "/"
# 自作モジュールのimport
sys.path.append(project_dir + "./my_modules/")
import detected_processing
import learning

# 単発音のデータの長さ
data_len = 2048

# tensorflowで必要な変数を取得
x, p, t, loss, train_step, correct_prediction, accuracy, y = learning.learning_algorithm(tf, data_len)

# tfのセッション初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# `do_learning.py`で学習させたデータを取得
saver = tf.train.Saver()
saver.restore(sess, project_dir + "./model_data/model.ckpt")

chunk = 2**10
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# 実行時に引数に時間を指定していたらその時間分実行し、指定していなかったらデフォルトで100秒
if len(sys.argv) == 2:
    RECORD_SECONDS = int(sys.argv[1]) 
else:
    RECORD_SECONDS = 100

pa = pyaudio.PyAudio()

stream = pa.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    frames_per_buffer = chunk
)

# データを入れていく
# allの長さは20以上にならない
all = []

# tmpは常に同じ長さ
tmp = [False for i in range(0, 20)]

print('指パッチンの検出を始めます')
for i in range(0, int(RATE / chunk * RECORD_SECONDS)):

    data = stream.read(chunk)
    npData = np.frombuffer(data, dtype="int16") / 32768.0

    # npDataの中にthresoldより大きい数字があるかどうか
    threshold = 0.05
    isThresholdOver = False
    if max(npData) > 0.05:
        isThresholdOver = True

    tmp.append(isThresholdOver)
    tmp.pop(0)

    # 9,10, 11がのどれかがtrueで他がfalseだけなら反応
    # iが11まではallの長さが足りないためエラーになる。
    if sum(tmp[9: 11]) >= 1 and sum(tmp) <= 3 and i >= 12:

        # 単発音の部分を取得する
        big_point_data = all[-10:-8] 

        big_point_data = np.frombuffer(b''.join(big_point_data), dtype="int16") / 32768.0
        X = np.fft.fft(big_point_data)
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]

        # 指パッチンである確率を算出
        result = sess.run(p, feed_dict={x: np.array([amplitudeSpectrum])})

        print('これがフィンガースナップである確率確率>>' + str(result[0][0]))

        if(result[0] >= 0.5):
            print('これは指パッチンです\n')
        else: 
            print('これは指パッチンではないです\n')


        tmp = [False for i in range(0, 20)]

    all.append(data)

    # allに入れっぱなしだとメモリを食う気がしたので、20以上なら0番目を削除
    if len(all) >= 20:
        all.pop(0)

stream.close()
pa.terminate()
print('指パッチン検出を終了します。')


import pyaudio
import sys
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tensorflow as tf
import datetime

import sys
project_dir = "/Users/imajo/Desktop/dev/google-assistant-mac/finger-snap/"
sys.path.append(project_dir + "learning/")
import learning_algorithm

data_len = 2048
x, p, t, loss, train_step, correct_prediction, accuracy = learning_algorithm.double_layer(tf, data_len)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, project_dir + "./model_data/model.ckpt")


chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
# 検知する時間
RECORD_SECONDS = 20

pa = pyaudio.PyAudio()

stream = pa.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    frames_per_buffer = chunk
)

# データを入れていく
all = []

# tmpは常に同じ長さ
tmp = [False for i in range(0, 20)]

# 1回のループで
# tmpの最後に閾値より大きいか大きくないかをTrue or Falseで入れる
# tmpの最初をとる
for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
    # dataが(RECORD_SECONDS / 128)秒でのデータ
    # このままではバイナリーデータなので変換する必要がある
    # np.frombuffer(data, dtype="int16") / 32768.0

    data = stream.read(chunk)
    npData = np.frombuffer(data, dtype="int16") / 32768.0
    # 閾値
    threshold = 0.85

    # npDataの中にthresoldより大きい数字があるかどうか
    isThresholdOver = npData[npData >= threshold].sum() >= 1

    tmp.append(isThresholdOver)
    tmp.pop(0)

    # 9,10, 11がのどれかがtrueで他がfalseだけなら反応
    # なぜか最初の10回目に誤反応するため、12回目までは反応しないようにしておく
    if sum(tmp[9: 11]) >= 1 and sum(tmp) <= 3 and i >= 12:
        #print("単発音を認識しました。")

        big_point_data = all[-10:-8] # 取得するbyteデータ

        data = np.frombuffer(b''.join(big_point_data), dtype="int16") / 32768.0
        X = np.fft.fft(data)
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]

        # 確率を算出
        result = sess.run(p, feed_dict={x: np.array([amplitudeSpectrum])})
        #print(result)
        if(result[0] >= 0.5):
            print('これは指パッチンです\n')
        else: 
            print('これは指パッチンではないです\n')


        tmp = [False for i in range(0, 20)]
        #try:
        #    with urllib.request.urlopen('http://localhost:8000/') as response:
        #       html = response.read()
        #except:
        #    print('get時のエラー')

    all.append(data)

stream.close()
p.terminate()


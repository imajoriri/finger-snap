import pyaudio
import sys
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tensorflow as tf
import datetime

import wave_sound

# original modules
import detected_processing
import sys
project_dir = "/Users/imajo/Desktop/dev/google-assistant-mac/finger-snap/"
sys.path.append(project_dir + "learning/")
import learning_algorithm

data_len = 2048
x, p, t, loss, train_step, correct_prediction, accuracy, y = learning_algorithm.double_layer(tf, data_len)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, project_dir + "./model_data/model.ckpt")

#chunk = 1024
chunk = 2**10
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
# 検知する時間
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
all = []

# tmpは常に同じ長さ
tmp = [False for i in range(0, 20)]

print('指パッチンの検出を始めます')
for i in range(0, int(RATE / chunk * RECORD_SECONDS)):

    data = stream.read(chunk)
    npData = np.frombuffer(data, dtype="int16") / 32768.0
    # 閾値
    threshold = 0.3

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

        #wave_sound.play_wave("./web_text_api/detection.wav")

        #fs = 44100
        #N = chunk * len(big_point_data) # FFTのサンプル数
        #d = 1.0/fs
        #freqList = np.fft.fftfreq(N, d) # (FFTのサンプル数(2**n), 1.0/fs) >> fsはサンプリングレート
        #plot_X(freqList, fs)
        #plt.show()
        #print("フィンガースナップを検出しました。")

        # 確率を算出
        result = sess.run(p, feed_dict={x: np.array([amplitudeSpectrum])})
        print('これがフィンガースナップである確率確率>>' + str(result[0][0]))
        if(result[0] >= 0.5):
            print('これは指パッチンです\n')
            #detected_processing.do_get('http://localhost')
            detected_processing.change_my_room_color()
            #wave_sound.play_wave("./web_text_api/isFinger.wav")
        else: 
            print('これは指パッチンではないです\n')
            #wave_sound.play_wave("./web_text_api/isNotFinger.wav")


        tmp = [False for i in range(0, 20)]
    all.append(data)

stream.close()
p.terminate()


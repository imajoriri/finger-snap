"""
データを保存するファイル
単発音を検知したら、/sounds/に保存して行く
以下で実行。

$ python input_data.py

"""

import pyaudio
import sys
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

def plot_x(x, N):
    # 波形を描画
    plt.subplot(311)  # 3行1列のグラフの1番目の位置にプロット
    plt.plot(range(N), x)
    plt.axis([0, N, -1.0, 1.0])
    plt.xlabel("time [sample]")
    plt.ylabel("amplitude")

def plot_X(freqList, fs):
    plt.subplot(312)  # 3行1列のグラフの1番目の位置にプロット
    plt.plot(freqList, amplitudeSpectrum, marker= 'o', linestyle='-')
    plt.axis([0, fs/2, 0, 50])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude spectrum")

chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
# 検知する時間
RECORD_SECONDS = 50

p = pyaudio.PyAudio()

stream = p.open(
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

print("検出を始めます。" + str(RECORD_SECONDS) + "秒間です")
for i in range(0, int(RATE / chunk * RECORD_SECONDS)):

    byte_data = stream.read(chunk) # len >> 2048
    int_data = np.frombuffer(byte_data, dtype="int16") / 32768.0 # len >> 1024
    all.append(byte_data)

    # npDataの中にthresoldより大きい数字があるかどうか
    threshold = 0.05
    isThresholdOver = False
    if max(int_data) > 0.05:
        isThresholdOver = True

    tmp.append(isThresholdOver)
    tmp.pop(0)

    # 9,10, 11がのどれかがtrueで他がfalseだけなら反応
    if sum(tmp[9: 11]) >= 1 and sum(tmp) <= 3 and i >= 12:
        print("単発音を認識しました。")

        big_point_data = all[-10:-8] # 取得するbyteデータ

        # 以下は固定
        fs = 44100
        N = chunk * len(big_point_data) # FFTのサンプル数
        d = 1.0/fs

        x = np.frombuffer(b''.join(big_point_data), dtype="int16") / 32768.0
        X = np.fft.fft(x)
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
        freqList = np.fft.fftfreq(N, d) # (FFTのサンプル数(2**n), 1.0/fs) >> fsはサンプリングレート

        # プロット
        #plot_x(x, N)
        #plot_X(freqList, fs)
        #plt.show()

        #plt.show()

        # 検出した部分をwavファイルで保存
        now = datetime.datetime.now()

        isFinger = input("finger:1 \nnot finger: 2 \n>> ")
        project_dir = os.getcwd() + "/"
        if isFinger == "1":
            file_name = project_dir + 'sounds/finger/{0:%Y%m%d%H%M%S}.wav'.format(now)
        elif isFinger == "2":
            file_name = project_dir + 'sounds/not-finger/{0:%Y%m%d%H%M%S}.wav'.format(now)
        else:
            print("1 or 2を入力してください")
            break

        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(big_point_data))
        wf.close()

        tmp = [False for i in range(0, 20)]
        print("検出を終了します")
        break

stream.close()
p.terminate()


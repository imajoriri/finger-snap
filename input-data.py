import pyaudio
import sys
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import datetime

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
RECORD_SECONDS = 3

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

for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
    # dataが(RECORD_SECONDS / 128)秒でのデータ
    # このままではバイナリーデータなので変換する必要がある
    # np.frombuffer(data, dtype="int16") / 32768.0

    byte_data = stream.read(chunk) # len >> 2048
    int_data = np.frombuffer(byte_data, dtype="int16") / 32768.0 # len >> 1024
    all.append(byte_data)

    # 閾値
    threshold = 0.9

    # npDataの中にthresoldより大きい数字があるかどうか
    isThresholdOver = int_data[int_data >= threshold].sum() >= 1

    tmp.append(isThresholdOver)
    tmp.pop(0)

    # 9,10, 11がのどれかがtrueで他がfalseだけなら反応
    # なぜか最初の10回目に誤反応するため、12回目までは反応しないようにしておく
    if sum(tmp[9: 11]) >= 1 and sum(tmp) <= 3 and i >= 12:
        print("フィンガースナップを認識しました。")

        # 単発音を検出したあたりのデータをフーリエ変換している。
        # フーリエ変換するデータの値の範囲を変更するならば >> all[-10:-8]
        # サンプル数(N)を操作するならば、入力する音声のプロットを変更する必要があるので >> chunk

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

        # 検出した部分をwavファイルで保存
        now = datetime.datetime.now()
        file_name = 'sounds/{0:%Y%m%d%H%M%S}.wav'.format(now)
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(big_point_data))
        wf.close()

        tmp = [False for i in range(0, 20)]
#        try:
#            with urllib.request.urlopen('http://localhost:8000/') as response:
#               html = response.read()
#        except:
#            print('get時のエラー')


stream.close()
p.terminate()

data = b''.join(all)

#x = np.frombuffer(data, dtype="int16") / 32768.0
## x.shape >> (132096,) *3秒の時
#
#fs = 44100
#d = 1.0/fs
#start = 1000
#N = 80000 # FFTのサンプル数
#freqList = np.fft.fftfreq(N, d) # (FFTのサンプル数(2**n), 1.0/fs) >> fsはサンプリングレート
#
## 高速フーリエ変換
#X = np.fft.fft(x[start:start + N])
#print("-- finish fft---")
#
#amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]



import pyaudio
import sys
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
# 検知する時間
RECORD_SECONDS = 10

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
    threshold = 0.9

    # npDataの中にthresoldより大きい数字があるかどうか
    isThresholdOver = npData[npData >= threshold].sum() >= 1

    tmp.append(isThresholdOver)
    tmp.pop(0)

    # 9,10, 11がのどれかがtrueで他がfalseだけなら反応
    # なぜか最初の10回目に誤反応するため、12回目までは反応しないようにしておく
    if sum(tmp[9: 11]) >= 1 and sum(tmp) <= 3 and i >= 12:
        print("フィンガースナップを認識しました。")
        tmp = [False for i in range(0, 20)]
        try:
            with urllib.request.urlopen('http://localhost:8000/') as response:
               html = response.read()
        except:
            print('get時のエラー')

    all.append(data)

stream.close()
p.terminate()

#data = b''.join(all)
#
#x = np.frombuffer(data, dtype="int16") / 32768.0
#
#plt.figure(figsize=(15,3))
#plt.plot(x)
#plt.show()

# 高速フーリエ変換
#fft = np.fft.fft(np.frombuffer(data, dtype="int16"))
#
#plt.figure(figsize=(15,3))
#plt.plot(fft.real[:int(len(x)/2)])
#plt.show()

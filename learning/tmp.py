# wavファイルを読み込んでフーリエ変換するまで

import pyaudio
import sys
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import datetime


def fft(file_name):
    #file_name = "./sounds/finger/20181007214123.wav"
    
    wave_file = wave.open(file_name, "r")
    
    buf = wave_file.readframes(wave_file.getnframes())
    # buf = waveFile.readframes(-1) # 全て読み込む場合
    wave_file.close()
    
    #x = np.frombuffer(b''.join(buf), dtype="int16") / 32768.0
    x = np.frombuffer(buf, dtype="int16") / 32768.0
    X = np.fft.fft(x)
    amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
    fs = 44100
    N = int(len(buf) / 2) # FFTのサンプル数
    d = 1.0/fs
    freqList = np.fft.fftfreq(N, d) # (FFTのサンプル数(2**n), 1.0/fs) >> fsはサンプリングレート
    
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
    
    plot_X(freqList, fs)
    plot_x(x, N)
    plt.show()



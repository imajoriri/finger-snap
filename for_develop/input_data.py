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

project_dir = os.getcwd() + "/"
# 自作モジュールのimport
sys.path.append(project_dir + "./my_modules/")
import const
import plot

# 検知する時間[秒]
RECORD_SECONDS = 50

p = pyaudio.PyAudio()

stream = p.open(
    format = const.FOR_PYAUDIO.FORMAT,
    channels = const.FOR_PYAUDIO.CHANNELS,
    rate = const.FOR_PYAUDIO.RATE,
    input = True,
    frames_per_buffer = const.FOR_PYAUDIO.chunk
)

def main():

    # 入力されたbyteデータを入れていく
    all = []
    
    # tmpは常に同じ長さ
    tmp = [False for i in range(0, 20)]
    
    print("検出を始めます。" + str(RECORD_SECONDS) + "秒間です")
    for i in range(0, int(const.FOR_PYAUDIO.RATE / const.FOR_PYAUDIO.chunk * RECORD_SECONDS)):
    
        byte_data = stream.read(const.FOR_PYAUDIO.chunk) # len >> 2048
        int_data = np.frombuffer(byte_data, dtype="int16") / 32768.0 # len >> 1024
        all.append(byte_data)
    
        # npDataの中にthresoldより大きい数字があれば、isThreshouldOverをTrueにする
        threshold = 0.05
        isThresholdOver = False
        if max(int_data) > threshold:
            isThresholdOver = True
    
        tmp.append(isThresholdOver)
        tmp.pop(0)
    
        # 9,10, 11がのどれかがtrueで他がfalseだけなら反応
        if sum(tmp[9: 11]) >= 1 and sum(tmp) <= 3 and i >= 12:
            print("単発音を認識しました。")
    
            big_point_data = all[-10:-8] # 取得するbyteデータ

            fs = const.FOR_PYAUDIO.chunk
            N = const.FOR_PYAUDIO.chunk * len(big_point_data) # FFTのサンプル数
            #N = fs * len(all[-17:-1]) # FFTのサンプル数
            d = 1.0/fs
    
            x = np.frombuffer(b''.join(big_point_data), dtype="int16") / 32768.0
            #x = np.frombuffer(b''.join(all[-17:-1]), dtype="int16") / 32768.0 * 1.8
            X = np.fft.fft(x)
            amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
            freqList = np.fft.fftfreq(N, d) # (FFTのサンプル数(2**n), 1.0/fs) >> fsはサンプリングレート

    
            # プロット
            #plot.plot_x(x, N)
            #plot.plot_X(freqList, amplitudeSpectrum, fs)
            #plt.show()
    
            # 検出した部分をwavファイルで保存
            now = datetime.datetime.now()
    
            # コマンドラインで入力を要求する
            # 1 >> 指パッチンとして保存
            # 2 >> 指パッチン以外のデータとして保存
            # (1 or 2)以外 >> 何もしない
            isFinger = input("finger:1 \nnot finger: 2 \n>> ")
    
            project_dir = os.getcwd() + "/"
    
            if isFinger == "1":
                if const.FOR_PYAUDIO.RATE == 44100:
                    file_name = project_dir + 'sounds/finger-44100/{0:%Y%m%d%H%M%S}.wav'.format(now)
                elif const.FOR_PYAUDIO.RATE == 16000:
                    file_name = project_dir + 'sounds/finger-16000/{0:%Y%m%d%H%M%S}.wav'.format(now)
    
            elif isFinger == "2":
                if const.FOR_PYAUDIO.RATE == 44100:
                    file_name = project_dir + 'sounds/not-finger-44100/{0:%Y%m%d%H%M%S}.wav'.format(now)
                elif const.FOR_PYAUDIO.RATE == 16000:
                    file_name = project_dir + 'sounds/not-finger-16000/{0:%Y%m%d%H%M%S}.wav'.format(now)
    
            else:
                print("1 or 2を入力してください")
                break

            # ./sounds ディレクトリが存在しなかったら作成する
            if os.path.isdir(project_dir + 'sounds') == False:
                os.mkdir(project_dir + "sounds/finger-16000")
                os.mkdir(project_dir + "sounds/finger-44100")
                os.mkdir(project_dir + "sounds/not-finger-16000")
                os.mkdir(project_dir + "sounds/not-finger-44100")
    
            # ファイル名をfile_nameとして保存
            wf = wave.open(file_name, 'wb')
            wf.setnchannels(const.FOR_PYAUDIO.CHANNELS)
            wf.setsampwidth(p.get_sample_size(const.FOR_PYAUDIO.FORMAT))
            wf.setframerate(const.FOR_PYAUDIO.RATE)
            wf.writeframes(b''.join(big_point_data))
            wf.close()
    
            tmp = [False for i in range(0, 20)]
            print("検出を終了します")
            break
    
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()

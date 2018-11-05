## -*- coding: utf-8 -*
#import pyaudio
#import wave 
#
#def main():
#    audio = pyaudio.PyAudio()
#
#    # 音声デバイス毎のインデックス番号を一覧表示
#    for x in range(0, audio.get_device_count()): 
#        print(audio.get_device_info_by_index(x))
#
#if __name__ == '__main__':
#    main()

# pyaudioをインポート
#import pyaudio
#
## PyAudioクラスのインスタンスを取得
#a = pyaudio.PyAudio()
#
## ホストAPI数
#hostAPICount = a.get_host_api_count()
#print("Host API Count = " + str(hostAPICount))
#
## ホストAPIの情報を列挙
#for cnt in range(0, hostAPICount):
#    print(a.get_host_api_info_by_index(cnt))
#
# ASIOデバイスの情報を列挙
#asioInfo = a.get_host_api_info_by_type(pyaudio.paASIO)
#print("ASIO Device Count = " + str(asioInfo.get("deviceCount")))
#for cnt in range(0, asioInfo.get("deviceCount")):
#    print(a.get_device_info_by_host_api_device_index(asioInfo.get("index"), cnt))

# -*- coding: utf-8 -*-
import sys
import pyaudio

# インデックス番号の確認

p = pyaudio.PyAudio()
count = p.get_device_count()
devices = []
for i in range(count):
    devices.append(p.get_device_info_by_index(i))

for i, dev in enumerate(devices):
    print (i, dev['name'])

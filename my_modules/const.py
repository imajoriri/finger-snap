import pyaudio

class FOR_PYAUDIO:
    # -- pyaudio関連---
    chunk = 2**10
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    #RATE = 44100
    RATE = 16000

class FOR_TENSORFLOW:
    DATA_LEN = 2048


# wave関連
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame, Series
import os
import wave
import random


def wave_fft(file_name):
    wave_file = wave.open(file_name, "r")
    buf = wave_file.readframes(wave_file.getnframes())
    wave_file.close()
    x = np.frombuffer(buf, dtype="int16") / 32768.0
    X = np.fft.fft(x)
    amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
    return amplitudeSpectrum

def add_train_data(dir, file_names, train_x, train_t, t):
    for file_name in file_names:
        file_name = dir + file_name
        amplitudeSpectrum = wave_fft(file_name)
        # ランダムで追加
        random_index = random.randrange(train_x.shape[0] + 1)
        train_x = np.insert(train_x, random_index, np.array([amplitudeSpectrum]), axis=0)
        train_t = np.insert(train_t, random_index, t)

    return train_x, train_t



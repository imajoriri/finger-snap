import matplotlib.pyplot as plt

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



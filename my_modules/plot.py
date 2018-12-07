import matplotlib.pyplot as plt

def plot_x(x, N):
    # 波形を描画
    plt.subplot(311)  # 3行1列のグラフの1番目の位置にプロット
    plt.plot(range(N), x, color="#85F768")
    plt.axis([0, N, -2.0, 2.0])
    plt.xlabel("time [sample]")
    plt.ylabel("amplitude")

def plot_X(freqList,amplitudeSpectrum, fs):
    plt.subplot(312)  # 3行1列のグラフの1番目の位置にプロット
    plt.plot(freqList, amplitudeSpectrum, linestyle='-')
    plt.axis([0, fs/4, 0, 70])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude spectrum")



# このアプリケーションについて

指パッチンを検出し、トリガーとして使用することができます。
それは、指パッチン一つで「非モテ男子」が一瞬にして「モテ男子」に変わります。。
実際に私が開発した、指パッチンをトリガーとしたものたちを紹介します。
***実際に「モテ男子」になる保証はしておりません。***

## Google Assistant

指パッチンでGoogleAssistantを起動します。

https://t.co/6SyPTS70xw

## 部屋の電気操作

指パッチンで部屋の電気を色っぽくします。

https://t.co/CiFGXvl39I

# 仕組み

どういう仕組みで動いているか、コード無しで説明した記事になります。

[非エンジニアでもわかる指パッチンで電球を色っぽくした話](https://qiita.com/imajoriri/items/0d1120917714bc4f3727)

# 環境

以下の環境で動作を確認済みです。

* Mac OS X(10.14.1)
* python3.5.1

# インストール

## pyaudioのインストール

pythonで音声データを扱うために`pyaudio`を使用します。
以下などを参考にインストールしてください。

[macOSにpyaudioをインストールする](https://qiita.com/mayfair/items/abb59ebf503cc294a581)

## そのほか必要なモジュールのインストール

必要なモジュールをインストールします。

```
$ pip install numpy numpy matplotlib tensorflow wave 
```

## このアプリケーションのインストール

`git clone`でローカルにダウンロードします。

```
$ git clone https://github.com/imajoriri/finger-snap.git
```

# 使い方

```
$ python detection_only.py 100

指パッチンの検出を始めます
```

`100`の部分は検出する時間[秒]です。  
指パッチンを検出すると、以下の表示が出ます。

```
これは指パッチンです
```

また、指パッチン以外の単発音を検出した場合は以下の表示が出ます。

```
これは指パッチンではないです
```



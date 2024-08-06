import numpy as np
import pandas as pd

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def bollinger_bands(series, window=20, window_dev=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    high_band = sma + (std * window_dev)
    low_band = sma - (std * window_dev)
    return high_band, low_band, sma

def rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def kama(series, span=10):
    return series.ewm(span=span, adjust=False).mean()

def ppo(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    ppo_line = ((exp1 - exp2) / exp2) * 100
    signal_line = ppo_line.ewm(span=signal, adjust=False).mean()
    return ppo_line, signal_line

def atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    return true_range.rolling(window=window).mean()

def aroon(series, window=25):
    aroon_up = series.rolling(window=window).apply(lambda x: x.argmax() / window * 100)
    aroon_down = series.rolling(window=window).apply(lambda x: x.argmin() / window * 100)
    aroon_indicator = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_indicator

def ichimoku(high, low, close):
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    chikou_span = close.shift(-26)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

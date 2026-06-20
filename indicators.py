"""
indicators.py
Modul indikator teknikal standar untuk analisis saham.
Semua fungsi menerima DataFrame dengan kolom: Open, High, Low, Close, Volume
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)


def volume_avg_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Rasio volume hari ini vs rata-rata volume - sinyal aktivitas tidak biasa."""
    avg_vol = volume.rolling(window=window).mean()
    return volume / avg_vol.replace(0, np.nan)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan semua indikator standar ke DataFrame OHLCV."""
    out = df.copy()
    out["SMA20"] = sma(out["Close"], 20)
    out["SMA50"] = sma(out["Close"], 50)
    out["SMA200"] = sma(out["Close"], 200)
    out["EMA12"] = ema(out["Close"], 12)
    out["EMA26"] = ema(out["Close"], 26)
    out["RSI14"] = rsi(out["Close"], 14)
    macd_line, signal_line, hist = macd(out["Close"])
    out["MACD"] = macd_line
    out["MACD_signal"] = signal_line
    out["MACD_hist"] = hist
    bb_u, bb_m, bb_l = bollinger_bands(out["Close"])
    out["BB_upper"] = bb_u
    out["BB_mid"] = bb_m
    out["BB_lower"] = bb_l
    out["ATR14"] = atr(out, 14)
    k, d = stochastic(out)
    out["Stoch_K"] = k
    out["Stoch_D"] = d
    out["Vol_Ratio"] = volume_avg_ratio(out["Volume"])
    return out

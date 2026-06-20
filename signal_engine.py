"""
signal_engine.py
Mesin penggabung sinyal -> menghasilkan keputusan BUY/SELL/HOLD dengan confidence score.

FILOSOFI BOBOT (penting, bisa Anda ubah sesuai keyakinan Anda):
- Teknikal (RSI, MACD, MA, Bollinger, Volume) : bobot besar -> dasar matematis & dipakai luas pasar
- ML model (klasifikasi arah dari indikator)   : bobot besar -> dilatih dari data historis aktual
- Fibonacci levels (support/resistance)         : bobot menengah -> level psikologis pasar
- Astro layer (moon/mercury)                    : bobot SANGAT KECIL & default OFF
                                                   -> non-empirical, hanya pelengkap opsional

Skor akhir dalam rentang -100 (Strong Sell) s.d. +100 (Strong Buy).
"""

import pandas as pd
import numpy as np
import indicators as ind
import fibonacci_astro as fa


DEFAULT_WEIGHTS = {
    "trend_ma": 20,        # posisi harga vs SMA20/50/200
    "rsi": 15,
    "macd": 20,
    "bollinger": 10,
    "volume": 10,
    "stochastic": 10,
    "fibonacci": 10,
    "ml_model": 25,         # ditambahkan terpisah di app.py setelah model load
    "astro": 0,             # default 0 = tidak mempengaruhi skor sama sekali
}


def _score_trend_ma(row) -> float:
    score = 0
    if pd.notna(row.get("SMA20")) and pd.notna(row.get("SMA50")):
        if row["Close"] > row["SMA20"] > row["SMA50"]:
            score += 1.0
        elif row["Close"] < row["SMA20"] < row["SMA50"]:
            score -= 1.0
        elif row["Close"] > row["SMA20"]:
            score += 0.4
        elif row["Close"] < row["SMA20"]:
            score -= 0.4
    if pd.notna(row.get("SMA200")):
        score += 0.3 if row["Close"] > row["SMA200"] else -0.3
    return np.clip(score, -1, 1)


def _score_rsi(row) -> float:
    rsi_val = row.get("RSI14", 50)
    if pd.isna(rsi_val):
        return 0
    if rsi_val < 30:
        return 1.0      # oversold -> potensi rebound (buy signal)
    elif rsi_val > 70:
        return -1.0     # overbought -> potensi koreksi (sell signal)
    else:
        return (50 - rsi_val) / 50 * 0.4  # netral, sedikit condong


def _score_macd(row) -> float:
    if pd.isna(row.get("MACD_hist")):
        return 0
    hist = row["MACD_hist"]
    macd_line = row.get("MACD", 0)
    signal_line = row.get("MACD_signal", 0)
    score = 0
    if macd_line > signal_line:
        score += 0.6
    else:
        score -= 0.6
    if hist > 0:
        score += 0.4
    else:
        score -= 0.4
    return np.clip(score, -1, 1)


def _score_bollinger(row) -> float:
    if pd.isna(row.get("BB_upper")) or pd.isna(row.get("BB_lower")):
        return 0
    close, upper, lower, mid = row["Close"], row["BB_upper"], row["BB_lower"], row["BB_mid"]
    band_width = upper - lower
    if band_width == 0:
        return 0
    position = (close - lower) / band_width  # 0 = di lower band, 1 = di upper band
    if position < 0.1:
        return 1.0
    elif position > 0.9:
        return -1.0
    else:
        return (0.5 - position) * 1.2


def _score_volume(row) -> float:
    vol_ratio = row.get("Vol_Ratio", 1.0)
    if pd.isna(vol_ratio):
        return 0
    price_change = row.get("Close", 0) - row.get("Open", 0)
    if vol_ratio > 1.5:
        return 0.8 if price_change > 0 else -0.8
    return 0


def _score_stochastic(row) -> float:
    k = row.get("Stoch_K", 50)
    if pd.isna(k):
        return 0
    if k < 20:
        return 1.0
    elif k > 80:
        return -1.0
    return 0


def _score_fibonacci(close: float, fib_levels: dict) -> float:
    """Skor berdasarkan kedekatan harga ke level fib 0.618/0.5 (area support kuat klasik)."""
    levels = fib_levels.get("levels", {})
    if not levels:
        return 0
    swing_high = fib_levels.get("swing_high", close)
    swing_low = fib_levels.get("swing_low", close)
    rng = swing_high - swing_low
    if rng == 0:
        return 0
    closest_dist = min(abs(close - v) for v in levels.values())
    proximity = 1 - min(closest_dist / (rng * 0.05), 1)  # dalam radius 5% dari range
    near_support = close <= (swing_low + rng * 0.5)
    return proximity * (0.8 if near_support else -0.3)


def generate_signal(df_with_indicators: pd.DataFrame, ml_score: float = None,
                     weights: dict = None, include_astro: bool = False,
                     astro_score: float = 0) -> dict:
    """
    df_with_indicators: hasil dari indicators.add_all_indicators()
    ml_score: skor dari ml_predictor.py, rentang -1 (sell) s.d +1 (buy). None jika belum ada model.
    weights: override DEFAULT_WEIGHTS jika perlu
    include_astro: apakah astro layer dimasukkan ke skor (default False - hanya informasi)
    """
    w = weights or DEFAULT_WEIGHTS.copy()
    if not include_astro:
        w["astro"] = 0

    row = df_with_indicators.iloc[-1]
    fib_levels = fa.fib_retracement(df_with_indicators)

    components = {
        "trend_ma": _score_trend_ma(row),
        "rsi": _score_rsi(row),
        "macd": _score_macd(row),
        "bollinger": _score_bollinger(row),
        "volume": _score_volume(row),
        "stochastic": _score_stochastic(row),
        "fibonacci": _score_fibonacci(float(row["Close"]), fib_levels),
        "astro": astro_score if include_astro else 0,
    }
    if ml_score is not None:
        components["ml_model"] = np.clip(ml_score, -1, 1)
    else:
        w = w.copy()
        w["ml_model"] = 0  # belum ada model -> bobot dialihkan ke teknikal

    total_weight = sum(w.values()) if sum(w.values()) > 0 else 1
    weighted_score = sum(components.get(k, 0) * w.get(k, 0) for k in w) / total_weight * 100

    if weighted_score >= 50:
        decision = "STRONG BUY"
    elif weighted_score >= 20:
        decision = "BUY"
    elif weighted_score <= -50:
        decision = "STRONG SELL"
    elif weighted_score <= -20:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {
        "decision": decision,
        "score": round(float(weighted_score), 1),
        "components": {k: round(float(v), 2) for k, v in components.items()},
        "weights_used": w,
        "fib_levels": fib_levels,
        "last_price": float(row["Close"]),
        "rsi": float(row.get("RSI14", np.nan)),
        "disclaimer": "Ini alat bantu analisis, BUKAN nasihat keuangan. Keputusan transaksi tetap tanggung jawab Anda sendiri."
    }

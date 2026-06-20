"""
data_fetcher.py
Modul pengambilan data harga saham IDX menggunakan yfinance (gratis, delay ~15-20 menit).

PENTING:
- Ticker IDX di Yahoo Finance pakai suffix ".JK" (contoh: BBCA.JK, TLKM.JK, BBRI.JK)
- Data ini TIDAK real-time. Ada delay ~15-20 menit dari harga aktual di RTI/broker.
- Jam bursa IDX (WIB): Sesi 1 = 09:00-11:30, Sesi 2 = 13:30-15:50 (Senin-Jumat)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

WIB = pytz.timezone("Asia/Jakarta")

# Daftar saham populer/likuid IDX untuk default watchlist
DEFAULT_WATCHLIST = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "UNVR.JK", "ICBP.JK", "ADRO.JK", "ANTM.JK",
    "GOTO.JK", "ARTO.JK", "MDKA.JK", "PGAS.JK", "INDF.JK"
]


def is_market_open():
    """Cek apakah pasar IDX sedang buka (estimasi, tidak memperhitungkan hari libur nasional)."""
    now = datetime.now(WIB)
    if now.weekday() >= 5:  # Sabtu/Minggu
        return False, "Pasar tutup (akhir pekan)"

    t = now.time()
    sesi1_start, sesi1_end = datetime.strptime("09:00", "%H:%M").time(), datetime.strptime("11:30", "%H:%M").time()
    sesi2_start, sesi2_end = datetime.strptime("13:30", "%H:%M").time(), datetime.strptime("15:50", "%H:%M").time()

    if sesi1_start <= t <= sesi1_end:
        return True, "Sesi 1 berjalan"
    elif sesi2_start <= t <= sesi2_end:
        return True, "Sesi 2 berjalan"
    elif sesi1_end < t < sesi2_start:
        return False, "Istirahat siang (jeda sesi 1 ke sesi 2)"
    else:
        return False, "Pasar tutup"


def normalize_ticker(ticker: str) -> str:
    """Pastikan ticker punya suffix .JK"""
    ticker = ticker.strip().upper()
    if not ticker.endswith(".JK"):
        ticker += ".JK"
    return ticker


def fetch_intraday(ticker: str, interval: str = "5m", period: str = "1d") -> pd.DataFrame:
    """
    Ambil data intraday untuk monitoring 'near-real-time'.
    interval: 1m, 2m, 5m, 15m, 30m, 60m
    period: berapa hari ke belakang (1d, 5d, 7d max untuk interval kecil)
    """
    ticker = normalize_ticker(ticker)
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_convert(WIB) if df.index.tz else df.index.tz_localize("UTC").tz_convert(WIB)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Ambil data harian untuk analisis swing/prediksi jangka menengah-panjang.
    period: 1y, 2y, 5y, max
    """
    ticker = normalize_ticker(ticker)
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_multiple_daily(tickers: list, period: str = "1y") -> dict:
    """Ambil data harian untuk banyak saham sekaligus (untuk screening bullish)."""
    result = {}
    for t in tickers:
        df = fetch_daily(t, period=period)
        if not df.empty:
            result[normalize_ticker(t)] = df
    return result


def get_last_price_info(ticker: str) -> dict:
    """Info harga terakhir + ARA/ARB check sederhana."""
    df = fetch_intraday(ticker, interval="5m", period="1d")
    if df.empty:
        df_daily = fetch_daily(ticker, period="5d")
        if df_daily.empty:
            return {}
        last = df_daily.iloc[-1]
        prev = df_daily.iloc[-2] if len(df_daily) > 1 else last
    else:
        last = df.iloc[-1]
        df_daily = fetch_daily(ticker, period="5d")
        prev = df_daily.iloc[-2] if len(df_daily) > 1 else last

    change = float(last["Close"]) - float(prev["Close"])
    change_pct = (change / float(prev["Close"])) * 100 if prev["Close"] != 0 else 0

    return {
        "ticker": normalize_ticker(ticker),
        "last_price": float(last["Close"]),
        "prev_close": float(prev["Close"]),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "volume": int(last["Volume"]) if not pd.isna(last["Volume"]) else 0,
        "timestamp": str(df.index[-1]) if not df.empty else "N/A (data harian)",
        "is_data_delayed": True,
    }


# Batas ARA/ARB IDX (auto rejection) berdasarkan rentang harga - aturan per Juni 2025
def get_ara_arb_limit(price: float) -> float:
    """
    Mengembalikan persentase batas auto rejection (ARA=atas, ARB=bawah) sesuai harga saham.
    Aturan BEI (perlu verifikasi ulang karena bisa berubah):
      - Rp 1 - Rp 200      : 35%
      - Rp 200 - Rp 5.000  : 25%
      - > Rp 5.000         : 20%
    """
    if price < 200:
        return 0.35
    elif price < 5000:
        return 0.25
    else:
        return 0.20

"""
fibonacci_astro.py
Modul analisis Fibonacci (valid secara teknikal) + lapisan siklus planet ala "Astronacci"
(DITANDAI: non-empirical / belief-based, bukan model yang divalidasi statistik).

PRINSIP KEJUJURAN METODOLOGIS:
- Fibonacci retracement/extension/time-zone = matematika murni, dipakai luas di analisis teknikal.
- Siklus planet (astro layer) = TIDAK punya dasar kausal yang terbukti secara ilmiah terhadap
  harga saham. Disertakan hanya karena diminta, dengan label tegas, dan diberi BOBOT KECIL
  dalam signal_engine.py supaya tidak mendominasi keputusan.
"""

import pandas as pd
import numpy as np
import ephem
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# 1. FIBONACCI (empirically-used technical tool, bukan hukum fisika tapi
#    levelnya dipakai luas oleh pelaku pasar -> punya "self-fulfilling" value)
# ---------------------------------------------------------------------------

FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXT_RATIOS = [1.272, 1.414, 1.618, 2.0, 2.618]


def fib_retracement(df: pd.DataFrame, lookback: int = 90) -> dict:
    """
    Hitung level retracement Fibonacci dari swing high/low dalam N hari terakhir.
    Cocok untuk menentukan area support/resistance potensial.
    """
    recent = df.tail(lookback)
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    diff = swing_high - swing_low

    levels = {}
    for r in FIB_RATIOS:
        levels[f"fib_{r}"] = swing_high - diff * r

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels": levels,
    }


def fib_extension(df: pd.DataFrame, lookback: int = 90) -> dict:
    """Level ekstensi Fibonacci untuk target profit/proyeksi harga di atas swing high."""
    base = fib_retracement(df, lookback)
    swing_high, swing_low = base["swing_high"], base["swing_low"]
    diff = swing_high - swing_low
    ext_levels = {f"ext_{r}": swing_low + diff * r for r in FIB_EXT_RATIOS}
    return ext_levels


def fib_time_zones(start_date, n_zones: int = 8) -> list:
    """
    Fibonacci time zone: tanggal-tanggal potensial titik balik berdasarkan urutan
    Fibonacci (1,1,2,3,5,8,13,21,...) dihitung dari hari dasar (swing low/high awal tren).
    Ini murni proyeksi waktu, bukan prediksi arah.
    """
    fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    dates = []
    for n in fib_seq[:n_zones]:
        dates.append(pd.Timestamp(start_date) + timedelta(days=int(n)))
    return dates


# ---------------------------------------------------------------------------
# 2. ASTRO LAYER (non-empirical) - siklus Bulan & posisi planet
#    Label "belief_based": True di setiap output supaya UI bisa menandainya.
# ---------------------------------------------------------------------------

def moon_phase(date: datetime) -> dict:
    """
    Fase bulan pada tanggal tertentu. Beberapa praktisi Astronacci mengasosiasikan
    new moon/full moon dengan potensi titik balik market (TIDAK TERVALIDASI SECARA STATISTIK).
    """
    obs_date = ephem.Date(date)
    nnm = ephem.next_new_moon(obs_date)
    pnm = ephem.previous_new_moon(obs_date)
    lunation = (obs_date - pnm) / (nnm - pnm)  # 0 = new moon, 0.5 = full moon, 1 = new moon lagi

    if lunation < 0.05 or lunation > 0.95:
        phase_name = "New Moon"
    elif 0.45 < lunation < 0.55:
        phase_name = "Full Moon"
    elif lunation < 0.5:
        phase_name = "Waxing"
    else:
        phase_name = "Waning"

    return {
        "date": str(date.date()),
        "lunation": round(lunation, 3),
        "phase_name": phase_name,
        "belief_based": True,
        "disclaimer": "Asosiasi fase bulan dengan titik balik harga TIDAK punya dasar empiris yang terbukti."
    }


def mercury_retrograde_check(date: datetime) -> dict:
    """
    Cek apakah Mercury sedang retrograde (gerak mundur semu dari Bumi).
    Beberapa praktisi astro-trading mengasosiasikan periode ini dengan volatilitas/kebingungan pasar.
    INI TIDAK ADA DASAR FISIKA/STATISTIK yang menghubungkannya dengan harga saham.
    """
    mercury = ephem.Mercury()
    d0 = ephem.Date(date - timedelta(days=1))
    d1 = ephem.Date(date)
    d2 = ephem.Date(date + timedelta(days=1))

    mercury.compute(d0)
    ra0 = mercury.ra
    mercury.compute(d2)
    ra2 = mercury.ra

    is_retrograde = ra2 < ra0  # right ascension menurun = retrograde (perkiraan kasar)

    return {
        "date": str(date.date()),
        "is_retrograde_estimate": bool(is_retrograde),
        "belief_based": True,
        "disclaimer": "Retrograde adalah fenomena optik astronomi murni (bukan fisik), tidak ada mekanisme kausal terhadap pasar saham."
    }


def get_astro_layer(date: datetime = None) -> dict:
    """Kumpulan sinyal astro untuk satu tanggal -> dipakai di signal_engine dgn bobot kecil."""
    if date is None:
        date = datetime.now()
    return {
        "moon": moon_phase(date),
        "mercury": mercury_retrograde_check(date),
        "note": "Layer ini bersifat belief-based / non-empirical. Bobot di sistem skoring sengaja dibuat kecil (lihat signal_engine.py)."
    }

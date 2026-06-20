"""
ml_predictor.py
Model machine learning untuk:
1. Klasifikasi arah pergerakan (naik/turun) jangka pendek -> dipakai signal_engine
2. Estimasi rentang harga (range) N hari & N bulan ke depan menggunakan volatilitas
   historis (model statistik) dikombinasikan dengan output klasifikasi ML (arah bias).

CATATAN JUJUR SOAL "ML KOMPLEKS":
Untuk MVP ini saya pakai Random Forest (model yang robust untuk data tabular kecil-menengah
dan tidak mudah overfit dibanding LSTM/Deep Learning pada data harian saham yang jumlahnya
terbatas, ~250-500 baris/tahun). LSTM/Deep Learning butuh ribuan-jutaan baris data untuk benar2
unggul, dan riset akademis (Bahkan dari OpsiQR, JP Morgan, dll) banyak menunjukkan model deep
learning TIDAK otomatis lebih baik dari Random Forest/gradient boosting untuk prediksi harga
saham harian dgn data sebatas IDX historis (~5-10 tahun). Jika Anda mau upgrade ke
LSTM nanti, strukturnya sudah modular (tinggal ganti class ini), tapi saya sarankan validasi
ekstra ketat (walk-forward backtest) sebelum dipakai untuk keputusan riil.

PERINGATAN PENTING: SEMUA prediksi harga saham punya batas akurasi fundamental karena pasar
dipengaruhi info baru yang tidak bisa diprediksi dari data historis saja (berita, sentimen,
kebijakan). Gunakan range, bukan angka pasti, dan SELALU sertakan confidence/uncertainty.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import indicators as ind

FEATURE_COLS = [
    "RSI14", "MACD", "MACD_hist", "Stoch_K", "Vol_Ratio",
    "SMA20", "SMA50", "BB_upper", "BB_lower",
]


def build_features(df_ind: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Siapkan fitur + target. Target = apakah harga naik > 1% dalam `horizon` hari ke depan.
    """
    df = df_ind.copy()
    df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target"] = (df["future_return"] > 0.01).astype(int)

    # fitur relatif (lebih stabil antar saham dibanding nilai absolut)
    df["price_vs_sma20"] = df["Close"] / df["SMA20"] - 1
    df["price_vs_sma50"] = df["Close"] / df["SMA50"] - 1
    df["bb_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)

    feature_cols = ["RSI14", "MACD_hist", "Stoch_K", "Vol_Ratio",
                     "price_vs_sma20", "price_vs_sma50", "bb_position"]
    df_feat = df[feature_cols + ["target", "Close"]].dropna()
    return df_feat, feature_cols


class DirectionPredictor:
    def __init__(self, horizon: int = 5, n_estimators: int = 200):
        self.horizon = horizon
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=6, min_samples_leaf=10,
            random_state=42, class_weight="balanced"
        )
        self.feature_cols = None
        self.is_fitted = False
        self.backtest_report = None

    def fit(self, df_ind: pd.DataFrame):
        df_feat, feature_cols = build_features(df_ind, self.horizon)
        self.feature_cols = feature_cols
        if len(df_feat) < 60:
            raise ValueError("Data historis terlalu sedikit untuk training model (butuh min ~60 baris valid).")

        X = df_feat[feature_cols].values
        y = df_feat["target"].values

        # Walk-forward backtest (TimeSeriesSplit) - WAJIB, jangan random split untuk data time series
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            temp_model = RandomForestClassifier(
                n_estimators=self.model.n_estimators, max_depth=6,
                min_samples_leaf=10, random_state=42, class_weight="balanced"
            )
            temp_model.fit(X_train, y_train)
            pred = temp_model.predict(X_test)
            accuracies.append(accuracy_score(y_test, pred))

        self.backtest_report = {
            "mean_accuracy": round(float(np.mean(accuracies)), 3),
            "fold_accuracies": [round(float(a), 3) for a in accuracies],
            "baseline_random": 0.5,
            "n_samples": len(df_feat),
            "note": "Akurasi >0.5 belum tentu signifikan secara statistik jika n_samples kecil. "
                    "Anggap model ini sebagai bias tambahan kecil, BUKAN sinyal pasti."
        }

        # fit final model dengan seluruh data
        self.model.fit(X, y)
        self.is_fitted = True
        return self.backtest_report

    def predict_score(self, df_ind: pd.DataFrame) -> float:
        """Skor -1 (turun) s.d +1 (naik) dari probabilitas model untuk baris terakhir."""
        if not self.is_fitted:
            return 0.0
        df_feat, _ = build_features(df_ind, self.horizon)
        if df_feat.empty:
            return 0.0
        last_row = df_ind.copy()
        last_row["price_vs_sma20"] = last_row["Close"] / last_row["SMA20"] - 1
        last_row["price_vs_sma50"] = last_row["Close"] / last_row["SMA50"] - 1
        last_row["bb_position"] = (last_row["Close"] - last_row["BB_lower"]) / (last_row["BB_upper"] - last_row["BB_lower"]).replace(0, np.nan)
        x_last = last_row[self.feature_cols].iloc[[-1]].fillna(0).values
        proba = self.model.predict_proba(x_last)[0]
        # proba[1] = probabilitas naik
        prob_up = proba[1] if len(proba) > 1 else 0.5
        return float((prob_up - 0.5) * 2)  # ubah ke skala -1..1


def estimate_price_range(df_ind: pd.DataFrame, n_days: int = 5, ml_bias_score: float = 0.0,
                          confidence: float = 0.80) -> dict:
    """
    Estimasi rentang harga N hari ke depan menggunakan model statistik:
    - Volatilitas historis (rolling std return harian) -> proyeksi via random walk + drift
    - ml_bias_score (dari DirectionPredictor) menggeser drift sedikit ke arah bias model

    Pendekatan ini LEBIH JUJUR daripada LSTM black-box untuk kasus data terbatas:
    rentang dihasilkan dari distribusi return historis aktual, bukan ekstrapolasi naif.
    """
    returns = df_ind["Close"].pct_change().dropna()
    if len(returns) < 30:
        return {"error": "Data historis tidak cukup untuk estimasi volatilitas yang andal."}

    daily_vol = float(returns.tail(252).std())  # vol 1 tahun terakhir
    daily_mean = float(returns.tail(252).mean())

    # drift disesuaikan sedikit oleh bias ML (dibatasi supaya tidak ekstrem)
    adjusted_drift = daily_mean + (ml_bias_score * daily_vol * 0.3)

    last_price = float(df_ind["Close"].iloc[-1])

    # proyeksi geometric random walk
    projected_mean = last_price * np.exp(adjusted_drift * n_days)
    projected_vol = daily_vol * np.sqrt(n_days)

    # z-score untuk confidence interval (pendekatan normal, asumsi disederhanakan)
    from scipy.stats import norm
    z = norm.ppf(0.5 + confidence / 2)

    lower = projected_mean * np.exp(-z * projected_vol)
    upper = projected_mean * np.exp(z * projected_vol)

    return {
        "n_days": n_days,
        "current_price": round(last_price, 2),
        "projected_price": round(float(projected_mean), 2),
        "range_lower": round(float(lower), 2),
        "range_upper": round(float(upper), 2),
        "confidence_level": confidence,
        "daily_volatility_pct": round(daily_vol * 100, 3),
        "ml_bias_applied": round(ml_bias_score, 3),
        "method": "Geometric Random Walk + historical volatility, drift disesuaikan ringan oleh skor ML",
        "disclaimer": "Estimasi statistik, BUKAN prediksi pasti. Rentang melebar drastis untuk horizon panjang -> wajar, mencerminkan ketidakpastian riil."
    }

"""
screener.py
Scan banyak saham sekaligus untuk menemukan kandidat yang paling "bullish" saat ini,
berdasarkan kombinasi skor teknikal (+ ML jika model tersedia).
"""

import pandas as pd
import numpy as np
import indicators as ind
import signal_engine as se
import ml_predictor as mlp
import data_fetcher as df_fetch
import sectors as sec


def screen_bullish_candidates(tickers: list, period: str = "1y", horizon: int = 5,
                                use_ml: bool = True, min_data_points: int = 100) -> pd.DataFrame:
    """
    Untuk setiap ticker: fetch data, hitung indikator, latih model ML cepat (per saham),
    hasilkan skor sinyal. Urutkan dari skor tertinggi (paling bullish) ke terendah.

    PERINGATAN: training model per-saham di sini cepat tapi sederhana (untuk screening cepat).
    Untuk analisis mendalam satu saham, gunakan modul ml_predictor langsung dengan tuning lebih baik.
    """
    results = []
    data_map = df_fetch.fetch_multiple_daily(tickers, period=period)

    for ticker, df in data_map.items():
        if len(df) < min_data_points:
            continue
        try:
            df_ind = ind.add_all_indicators(df)
            df_ind = df_ind.dropna(subset=["SMA50"])
            if len(df_ind) < min_data_points:
                continue

            ml_score = None
            backtest_acc = None
            if use_ml:
                try:
                    predictor = mlp.DirectionPredictor(horizon=horizon, n_estimators=100)
                    report = predictor.fit(df_ind)
                    ml_score = predictor.predict_score(df_ind)
                    backtest_acc = report["mean_accuracy"]
                except Exception:
                    ml_score = None

            signal = se.generate_signal(df_ind, ml_score=ml_score)
            range_est = mlp.estimate_price_range(df_ind, n_days=horizon, ml_bias_score=ml_score or 0)

            results.append({
                "ticker": ticker,
                "sector": sec.get_sector_label(ticker),
                "last_price": signal["last_price"],
                "decision": signal["decision"],
                "score": signal["score"],
                "rsi": round(signal["rsi"], 1) if not np.isnan(signal["rsi"]) else None,
                "ml_backtest_accuracy": backtest_acc,
                "projected_price": range_est.get("projected_price"),
                "range_lower": range_est.get("range_lower"),
                "range_upper": range_est.get("range_upper"),
            })
        except Exception as e:
            print(f"Skip {ticker}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return result_df


def top_n_per_sector(result_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Dari hasil screen_bullish_candidates() (sudah punya kolom 'sector'), ambil
    top-N saham paling bullish (skor tertinggi) UNTUK SETIAP SEKTOR secara terpisah.

    Tujuannya: hindari hasil yang didominasi 1-2 sektor saja (misal saat satu sektor
    sedang rally serentak, sektor lain yang sebenarnya juga punya kandidat bagus jadi
    'tertutup' di ranking global). Dengan ini, tiap sektor punya jagoannya sendiri.
    """
    if result_df.empty or "sector" not in result_df.columns:
        return result_df
    return (
        result_df.sort_values("score", ascending=False)
        .groupby("sector", group_keys=True)
        .head(top_n)
        .sort_values(["sector", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def sector_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ringkasan per sektor: skor rata-rata, skor maksimum (saham paling bullish di sektor itu),
    dan jumlah saham yang berhasil dianalisis. Berguna untuk lihat sektor mana yang
    'paling panas' secara keseluruhan, bukan cuma satu saham individu.
    """
    if result_df.empty or "sector" not in result_df.columns:
        return pd.DataFrame()
    summary = (
        result_df.groupby("sector")
        .agg(
            avg_score=("score", "mean"),
            max_score=("score", "max"),
            n_saham=("ticker", "count"),
        )
        .reset_index()
    )
    # ambil ticker dengan skor tertinggi di tiap sektor sebagai representasi "jagoan sektor"
    top_tickers = []
    for sector_name in summary["sector"]:
        sub = result_df[result_df["sector"] == sector_name]
        top_row = sub.loc[sub["score"].idxmax()]
        top_tickers.append(top_row["ticker"])
    summary["top_ticker"] = top_tickers
    summary["avg_score"] = summary["avg_score"].round(1)
    summary["max_score"] = summary["max_score"].round(1)
    summary = summary.sort_values("avg_score", ascending=False).reset_index(drop=True)
    return summary

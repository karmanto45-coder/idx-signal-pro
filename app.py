"""
app.py - IDX Signal Pro
Aplikasi analisis saham IDX: monitoring near-real-time, sinyal beli/jual,
dan prediksi rentang harga jangka pendek-panjang.

CARA JALANKAN LOKAL:
    streamlit run app.py

DEPLOY: Streamlit Community Cloud (gratis) - ikuti pola yang sama seperti SpectraVision Pro.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import data_fetcher as fetch
import indicators as ind
import fibonacci_astro as fa
import signal_engine as se
import ml_predictor as mlp
import screener as scr

st.set_page_config(page_title="IDX Signal Pro", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("📈 IDX Signal Pro")
st.sidebar.caption("Analisis saham IDX — personal tool, bukan nasihat keuangan resmi.")

is_open, market_status = fetch.is_market_open()
status_color = "🟢" if is_open else "🔴"
st.sidebar.markdown(f"**Status Pasar:** {status_color} {market_status}")
st.sidebar.caption(f"Waktu sekarang (WIB): {datetime.now().strftime('%H:%M:%S, %d %b %Y')}")
st.sidebar.warning("⚠️ Data harga memakai sumber gratis (Yahoo Finance) dengan delay ~15-20 menit. "
                    "Bukan data tick-by-tick real-time.")

page = st.sidebar.radio("Menu", [
    "🎯 Sinyal Beli/Jual (Per Saham)",
    "🔍 Screener Saham Bullish",
    "📊 Prediksi Rentang Harga",
])

st.sidebar.divider()
include_astro = st.sidebar.checkbox(
    "Aktifkan layer Astronacci (Fibonacci + siklus planet)",
    value=False,
    help="Komponen Fibonacci valid secara teknikal. Komponen siklus planet (bulan/Mercury retrograde) "
         "TIDAK punya dasar empiris yang terbukti -> bobotnya dibuat sangat kecil & default off."
)
astro_weight = 0
if include_astro:
    astro_weight = st.sidebar.slider("Bobot astro layer (%)", 0, 15, 5,
                                       help="Disarankan tetap kecil. Ini layer belief-based.")

st.sidebar.divider()
st.sidebar.caption("💡 Dibangun bersama Claude — iteratif, sesuai prinsip rigor yang sama dengan riset spektroskopi Anda.")


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_data(ticker, period="2y"):
    return fetch.fetch_daily(ticker, period=period)


def plot_candlestick_with_signals(df_ind, ticker, fib_levels=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                         row_heights=[0.55, 0.2, 0.25],
                         subplot_titles=(f"{ticker} - Price & MA", "RSI", "MACD"))

    fig.add_trace(go.Candlestick(
        x=df_ind.index, open=df_ind["Open"], high=df_ind["High"],
        low=df_ind["Low"], close=df_ind["Close"], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA20"], name="SMA20",
                              line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA50"], name="SMA50",
                              line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_upper"], name="BB Upper",
                              line=dict(width=1, dash="dot"), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_lower"], name="BB Lower",
                              line=dict(width=1, dash="dot"), opacity=0.5), row=1, col=1)

    if fib_levels:
        for name, level in fib_levels["levels"].items():
            fig.add_hline(y=level, line_dash="dash", line_color="orange", opacity=0.4,
                          annotation_text=name.replace("fib_", "Fib "), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI14"], name="RSI",
                              line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"], name="MACD",
                              line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_signal"], name="Signal",
                              line=dict(color="orange")), row=3, col=1)
    fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_hist"], name="Histogram",
                          marker_color="gray", opacity=0.5), row=3, col=1)

    fig.update_layout(height=750, xaxis_rangeslider_visible=False,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


# ---------------------------------------------------------------------------
# PAGE 1: SINYAL BELI/JUAL PER SAHAM
# ---------------------------------------------------------------------------
if page == "🎯 Sinyal Beli/Jual (Per Saham)":
    st.title("🎯 Sinyal Beli/Jual — Analisis Saham")

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker_input = st.selectbox(
            "Pilih atau ketik kode saham (tanpa .JK)",
            options=[t.replace(".JK", "") for t in fetch.DEFAULT_WATCHLIST],
            index=0,
        )
        custom_ticker = st.text_input("Atau ketik kode saham lain", "")
        ticker = custom_ticker if custom_ticker else ticker_input
    with col2:
        period = st.selectbox("Periode data historis", ["6mo", "1y", "2y", "5y"], index=1)
        horizon = st.number_input("Horizon prediksi ML (hari)", 1, 60, 5)

    if st.button("🔄 Analisis Sekarang", type="primary"):
        with st.spinner(f"Mengambil data {ticker}..."):
            df = load_data(ticker, period=period)

        if df.empty:
            st.error(f"Data untuk {ticker} tidak ditemukan. Pastikan kode saham benar (contoh: BBCA, TLKM).")
        else:
            df_ind = ind.add_all_indicators(df)

            with st.spinner("Melatih model ML & backtesting..."):
                try:
                    predictor = mlp.DirectionPredictor(horizon=int(horizon))
                    backtest = predictor.fit(df_ind)
                    ml_score = predictor.predict_score(df_ind)
                except Exception as e:
                    backtest = None
                    ml_score = None
                    st.warning(f"Model ML tidak bisa dilatih: {e}")

            astro_score = 0
            astro_info = None
            if include_astro:
                astro_info = fa.get_astro_layer(datetime.now())
                astro_score = 0.3 if astro_info["moon"]["phase_name"] == "New Moon" else 0

            weights = se.DEFAULT_WEIGHTS.copy()
            if include_astro:
                weights["astro"] = astro_weight

            signal = se.generate_signal(df_ind, ml_score=ml_score, weights=weights,
                                          include_astro=include_astro, astro_score=astro_score)

            # --- Header hasil ---
            decision_color = {
                "STRONG BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡",
                "SELL": "🔴", "STRONG SELL": "🔴🔴"
            }.get(signal["decision"], "⚪")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga Terakhir", f"Rp {signal['last_price']:,.0f}")
            c2.metric("Keputusan", f"{decision_color} {signal['decision']}")
            c3.metric("Skor Sinyal", f"{signal['score']}/100")
            c4.metric("RSI", f"{signal['rsi']:.1f}")

            st.caption(signal["disclaimer"])

            if backtest:
                with st.expander("📋 Validitas Model ML (Backtest)"):
                    st.json(backtest)
                    if backtest["mean_accuracy"] < 0.55:
                        st.error("Akurasi model mendekati tebakan acak (50%). Jangan andalkan komponen ML untuk saham ini.")
                    elif backtest["mean_accuracy"] < 0.6:
                        st.warning("Akurasi model rendah-menengah. Gunakan sebagai bias kecil saja, bukan sinyal utama.")
                    else:
                        st.success("Akurasi model di atas rata-rata, namun tetap perlu pengawasan berkelanjutan (model bisa usang).")

            # --- Breakdown komponen sinyal ---
            with st.expander("🔬 Rincian Komponen Sinyal", expanded=True):
                comp_df = pd.DataFrame([
                    {"Komponen": k, "Skor (-1 s.d 1)": v, "Bobot (%)": signal["weights_used"].get(k, 0)}
                    for k, v in signal["components"].items()
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            if include_astro and astro_info:
                with st.expander("🌙 Layer Astronacci (non-empirical, informasi tambahan saja)"):
                    st.json(astro_info)

            # --- Chart ---
            st.plotly_chart(plot_candlestick_with_signals(df_ind, ticker, signal["fib_levels"]),
                             use_container_width=True)

            # --- ARA/ARB info ---
            limit_pct = fetch.get_ara_arb_limit(signal["last_price"])
            ara_price = signal["last_price"] * (1 + limit_pct)
            arb_price = signal["last_price"] * (1 - limit_pct)
            st.info(f"📏 Estimasi batas ARA/ARB hari ini: **ARB Rp {arb_price:,.0f}** — "
                    f"**ARA Rp {ara_price:,.0f}** (±{limit_pct*100:.0f}%, perlu verifikasi aturan terbaru BEI)")


# ---------------------------------------------------------------------------
# PAGE 2: SCREENER BULLISH
# ---------------------------------------------------------------------------
elif page == "🔍 Screener Saham Bullish":
    st.title("🔍 Screener Saham Bullish")
    st.caption("Scan beberapa saham sekaligus, urutkan dari yang paling bullish berdasarkan skor gabungan.")

    default_list = ", ".join([t.replace(".JK", "") for t in fetch.DEFAULT_WATCHLIST])
    tickers_text = st.text_area("Daftar kode saham (pisahkan dengan koma)", default_list, height=80)
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        horizon_scr = st.number_input("Horizon (hari)", 1, 60, 5, key="scr_horizon")
    with col2:
        use_ml_scr = st.checkbox("Sertakan model ML (lebih lambat, lebih akurat)", value=True)

    st.warning(f"Akan menganalisis {len(tickers)} saham. Proses bisa memakan waktu beberapa menit jika ML diaktifkan.")

    if st.button("🚀 Jalankan Screening", type="primary"):
        progress = st.progress(0, text="Memulai screening...")
        with st.spinner("Mengambil & menganalisis data..."):
            result_df = scr.screen_bullish_candidates(tickers, horizon=int(horizon_scr), use_ml=use_ml_scr)
        progress.progress(100, text="Selesai")

        if result_df.empty:
            st.error("Tidak ada hasil. Cek kode saham yang dimasukkan.")
        else:
            st.success(f"Ditemukan {len(result_df)} saham dengan data valid.")
            st.dataframe(
                result_df.style.background_gradient(subset=["score"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True
            )
            st.caption("Sinyal ini alat bantu screening, bukan rekomendasi final. Selalu cek fundamental & berita terbaru sebelum transaksi.")

            top3 = result_df.head(3)
            st.subheader("🏆 Top 3 Kandidat Paling Bullish")
            for _, row in top3.iterrows():
                st.markdown(f"**{row['ticker']}** — {row['decision']} (skor {row['score']}) — "
                            f"Harga: Rp {row['last_price']:,.0f}, Proyeksi: Rp {row['projected_price']:,.0f} "
                            f"(range Rp {row['range_lower']:,.0f} - Rp {row['range_upper']:,.0f})")


# ---------------------------------------------------------------------------
# PAGE 3: PREDIKSI RENTANG HARGA
# ---------------------------------------------------------------------------
elif page == "📊 Prediksi Rentang Harga":
    st.title("📊 Prediksi Rentang Harga Jangka Pendek & Panjang")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_p = st.text_input("Kode saham (tanpa .JK)", "BBCA")
    with col2:
        n_days = st.number_input("N hari ke depan", 1, 90, 30)
    with col3:
        n_months_days = st.number_input("N bulan ke depan (dikonversi ke hari trading)", 1, 36, 6)

    confidence = st.slider("Tingkat keyakinan interval (%)", 50, 95, 80) / 100

    if st.button("📈 Hitung Prediksi", type="primary"):
        with st.spinner(f"Mengambil data {ticker_p}..."):
            df = load_data(ticker_p, period="5y")

        if df.empty:
            st.error("Data tidak ditemukan.")
        else:
            df_ind = ind.add_all_indicators(df)
            try:
                predictor = mlp.DirectionPredictor(horizon=int(n_days))
                backtest = predictor.fit(df_ind)
                ml_score = predictor.predict_score(df_ind)
            except Exception:
                backtest = None
                ml_score = 0

            range_days = mlp.estimate_price_range(df_ind, n_days=int(n_days),
                                                    ml_bias_score=ml_score or 0, confidence=confidence)
            trading_days_per_month = 21
            range_months = mlp.estimate_price_range(
                df_ind, n_days=int(n_months_days * trading_days_per_month),
                ml_bias_score=(ml_score or 0) * 0.5,  # bias ML dikurangi utk horizon panjang -> lebih tidak pasti
                confidence=confidence
            )

            st.subheader(f"Estimasi {n_days} Hari ke Depan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Harga Sekarang", f"Rp {range_days['current_price']:,.0f}")
            c2.metric("Proyeksi (titik tengah)", f"Rp {range_days['projected_price']:,.0f}")
            c3.metric(f"Rentang ({int(confidence*100)}% confidence)",
                      f"Rp {range_days['range_lower']:,.0f} - Rp {range_days['range_upper']:,.0f}")

            st.subheader(f"Estimasi {n_months_days} Bulan ke Depan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Harga Sekarang", f"Rp {range_months['current_price']:,.0f}")
            c2.metric("Proyeksi (titik tengah)", f"Rp {range_months['projected_price']:,.0f}")
            c3.metric(f"Rentang ({int(confidence*100)}% confidence)",
                      f"Rp {range_months['range_lower']:,.0f} - Rp {range_months['range_upper']:,.0f}")

            st.error("⚠️ **PENTING:** Rentang ini melebar jauh untuk horizon panjang — ini BUKAN bug, "
                     "tapi cerminan ketidakpastian riil pasar saham. Jangan pernah memperlakukan titik tengah "
                     "proyeksi sebagai target pasti. Semakin panjang horizon, semakin besar peran faktor "
                     "fundamental & berita yang TIDAK bisa ditangkap model historis ini.")

            if backtest:
                with st.expander("📋 Validitas Model (Backtest)"):
                    st.json(backtest)

            with st.expander("📐 Metodologi"):
                st.markdown(f"""
                - **Metode dasar:** Geometric Random Walk menggunakan volatilitas historis 1 tahun terakhir
                - **Bias arah:** disesuaikan ringan oleh skor model Random Forest (akurasi backtest: {backtest['mean_accuracy'] if backtest else 'N/A'})
                - **Mengapa bukan LSTM/Deep Learning murni?** Dengan data harian terbatas (~5 tahun = ~1250 baris),
                  model statistik berbasis volatilitas + bias ML ringan terbukti lebih robust & tidak overfitting
                  dibanding deep learning yang butuh data jauh lebih banyak untuk generalisasi baik.
                - **Interval kepercayaan** dihitung dari distribusi normal pada log-return (pendekatan, bukan distribusi sebenarnya
                  yang biasanya punya fat-tail / lebih ekstrem dari normal).
                """)

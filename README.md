# IDX Signal Pro

Aplikasi analisis saham IDX (Bursa Efek Indonesia) — personal tool untuk monitoring sinyal beli/jual
dan prediksi rentang harga jangka pendek-panjang.

## ⚠️ Batasan Penting (wajib dibaca)

1. **Bukan data real-time sesungguhnya.** Memakai Yahoo Finance (gratis), delay ~15-20 menit dari harga aktual di lantai bursa. Untuk data tick-by-tick asli, perlu API berbayar (Stockbit Pro, RTI Business, atau API broker resmi).
2. **Bukan nasihat keuangan.** Semua skor, sinyal, dan prediksi adalah alat bantu analisis berbasis data historis. Pasar saham dipengaruhi info baru (berita, kebijakan, sentimen) yang tidak bisa ditangkap model historis.
3. **Layer Astronacci (siklus planet) bersifat belief-based**, bukan tervalidasi secara statistik/ilmiah. Default OFF, dan jika diaktifkan bobotnya sengaja dibuat kecil. Komponen Fibonacci-nya sendiri (retracement/extension) adalah alat teknikal yang sah dipakai luas oleh pasar.
4. **Model ML bisa usang** (model drift). Selalu cek bagian "Validitas Model (Backtest)" — jika akurasi mendekati 50%, jangan andalkan komponen ML untuk saham tersebut.
5. **Aturan ARA/ARB di kode adalah estimasi** dan perlu diverifikasi ulang ke aturan BEI terkini karena bisa berubah.

## Struktur Proyek

```
IDX-Signal-Pro/
├── app.py                      # Aplikasi Streamlit utama (3 halaman)
├── modules/
│   ├── data_fetcher.py         # Ambil data via yfinance, cek jam bursa
│   ├── indicators.py           # RSI, MACD, SMA/EMA, Bollinger, ATR, Stochastic
│   ├── fibonacci_astro.py      # Fibonacci (valid) + astro layer (non-empirical, berlabel jelas)
│   ├── signal_engine.py        # Penggabung skor -> keputusan BUY/SELL/HOLD
│   ├── ml_predictor.py         # Random Forest klasifikasi arah + estimasi rentang harga
│   └── screener.py             # Scan banyak saham, urutkan paling bullish
├── requirements.txt
└── README.md
```

## Cara Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Deploy ke Streamlit Community Cloud (gratis)

1. Buat repo GitHub baru, upload semua file (struktur folder dipertahankan — `modules/` harus tetap subfolder).
2. Masuk ke [share.streamlit.io](https://share.streamlit.io), connect ke repo tersebut.
3. Set **Main file path**: `app.py`
4. Deploy. Streamlit Cloud otomatis install dari `requirements.txt`.

> Catatan: berbeda dari SpectraVision Pro yang butuh Supabase (karena perlu simpan data persisten antar user), aplikasi ini tidak butuh database — semua kalkulasi dilakukan on-the-fly dari data yfinance. Jika nanti mau menyimpan watchlist personal/histori sinyal, baru perlu tambah Supabase.

## Roadmap Pengembangan Selanjutnya (saran prioritas)

1. **Validasi backtest lebih ketat** — walk-forward testing dengan periode lebih panjang, per-sektor.
2. **Data broker/foreign flow** — butuh sumber data tambahan (biasanya berbayar) untuk sinyal yang lebih kuat dari sekadar indikator teknikal.
3. **Watchlist personal + notifikasi** — perlu database (Supabase) untuk simpan watchlist & alert harga.
4. **Auto-refresh saat market berjalan** — Streamlit punya `st.fragment` + `time.sleep` untuk auto-refresh tiap interval tertentu saat sesi bursa berjalan.
5. **Multi-timeframe confirmation** — gabungkan sinyal dari beberapa timeframe (intraday + harian + mingguan) untuk konfirmasi lebih kuat sebelum keputusan.

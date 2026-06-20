"""
sectors.py
Klasifikasi sektor saham IDX berdasarkan IDX-IC (IDX Industrial Classification),
diberlakukan BEI sejak 25 Januari 2021, menggantikan JASICA.

CATATAN PENTING SOAL AKURASI:
- IDX-IC resmi punya 12 sektor (termasuk sektor Z: Produk Investasi Tercatat, untuk
  ETF/produk investasi, bukan saham operasional) dengan total ~900+ saham, 35 sub-sektor,
  69 industri, 130 sub-industri.
- Modul ini HANYA mencakup ~80 saham paling likuid/blue-chip per sektor (kurasi manual),
  BUKAN seluruh universe BEI. Tujuannya untuk screening cepat & relevan, bukan riset
  kelengkapan sektor.
- Klasifikasi tiap ticker mengikuti sektor utama IDX-IC per pengetahuan terakhir penulis.
  Emiten bisa direklasifikasi BEI dari waktu ke waktu -> selalu verifikasi ulang via
  www.idx.co.id/data-pasar/data-saham/daftar-saham/ untuk keperluan riset/publikasi resmi.
"""

# 11 sektor operasional IDX-IC (kode huruf A-K resmi BEI)
SECTORS = {
    "A": {"name": "Energi", "emoji": "🛢️"},
    "B": {"name": "Barang Baku", "emoji": "⛏️"},
    "C": {"name": "Perindustrian", "emoji": "🏭"},
    "D": {"name": "Konsumer Primer", "emoji": "🛒"},
    "E": {"name": "Konsumer Non-Primer", "emoji": "👜"},
    "F": {"name": "Kesehatan", "emoji": "💊"},
    "G": {"name": "Keuangan", "emoji": "🏦"},
    "H": {"name": "Properti & Real Estat", "emoji": "🏢"},
    "I": {"name": "Teknologi", "emoji": "💻"},
    "J": {"name": "Infrastruktur", "emoji": "🏗️"},
    "K": {"name": "Transportasi & Logistik", "emoji": "🚚"},
}

# Ticker (tanpa .JK) -> kode sektor. Daftar likuid/blue-chip per sektor, kurasi manual.
TICKER_SECTOR = {
    # A - Energi
    "ADRO": "A", "PTBA": "A", "ITMG": "A", "MEDC": "A",
    "PGAS": "A", "AKRA": "A", "HRUM": "A", "INDY": "A",
    # B - Barang Baku
    "ANTM": "B", "INCO": "B", "TINS": "B", "MDKA": "B",
    "INKP": "B", "SMGR": "B", "INTP": "B", "BRMS": "B",
    # C - Perindustrian
    "ASII": "C", "UNTR": "C", "HEXA": "C", "IMAS": "C",
    # D - Konsumer Primer
    "INDF": "D", "ICBP": "D", "MYOR": "D", "GGRM": "D",
    "HMSP": "D", "CPIN": "D", "AALI": "D", "AMRT": "D",
    # E - Konsumer Non-Primer
    "ACES": "E", "MAPI": "E", "ERAA": "E", "AUTO": "E",
    "RALS": "E", "LPPF": "E", "MAPA": "E", "BMTR": "E",
    # F - Kesehatan
    "KLBF": "F", "KAEF": "F", "SIDO": "F", "SILO": "F",
    "MIKA": "F", "HEAL": "F", "PRDA": "F",
    # G - Keuangan
    "BBCA": "G", "BBRI": "G", "BMRI": "G", "BBNI": "G",
    "BBTN": "G", "ARTO": "G", "BRIS": "G", "PNBN": "G",
    # H - Properti & Real Estat
    "BSDE": "H", "PWON": "H", "CTRA": "H", "SMRA": "H",
    "PANI": "H", "LPKR": "H",
    # I - Teknologi
    "GOTO": "I", "BUKA": "I", "EMTK": "I", "WIRG": "I",
    "DCII": "I", "MTDL": "I",
    # J - Infrastruktur
    "TLKM": "J", "JSMR": "J", "EXCL": "J", "ISAT": "J",
    "TOWR": "J", "TBIG": "J", "PGEO": "J", "WIKA": "J",
    "PTPP": "J", "ADHI": "J",
    # K - Transportasi & Logistik
    "ASSA": "K", "BIRD": "K", "SMDR": "K", "TMAS": "K",
    "GIAA": "K", "WEHA": "K",
}


def _clean(ticker: str) -> str:
    return ticker.strip().upper().replace(".JK", "")


def get_sector_code(ticker: str) -> str:
    """Kode sektor (A-K) untuk sebuah ticker, atau '?' jika belum terklasifikasi di modul ini."""
    return TICKER_SECTOR.get(_clean(ticker), "?")


def get_sector_label(ticker: str) -> str:
    """Label sektor lengkap dengan emoji, contoh: '🏦 Keuangan'. 'Lainnya' jika tidak terklasifikasi."""
    code = get_sector_code(ticker)
    info = SECTORS.get(code)
    if not info:
        return "❓ Lainnya / Belum Terklasifikasi"
    return f"{info['emoji']} {info['name']}"


def get_sector_name(ticker: str) -> str:
    """Nama sektor tanpa emoji, untuk sorting/grouping yang konsisten."""
    code = get_sector_code(ticker)
    info = SECTORS.get(code)
    return info["name"] if info else "Lainnya / Belum Terklasifikasi"


def all_tickers(with_suffix: bool = True) -> list:
    """Semua ticker dalam watchlist kurasi ini, urut sesuai sektor."""
    ordered = sorted(TICKER_SECTOR.items(), key=lambda kv: (kv[1], kv[0]))
    tickers = [t for t, _ in ordered]
    return [f"{t}.JK" for t in tickers] if with_suffix else tickers


def tickers_by_sector(sector_codes=None, with_suffix: bool = True) -> dict:
    """
    Mapping {nama_sektor: [ticker, ...]} untuk kode sektor yang dipilih.
    sector_codes=None -> semua 11 sektor.
    """
    codes = sector_codes or list(SECTORS.keys())
    result = {}
    for code in codes:
        if code not in SECTORS:
            continue
        label = f"{SECTORS[code]['emoji']} {SECTORS[code]['name']}"
        members = [t for t, c in TICKER_SECTOR.items() if c == code]
        members.sort()
        if with_suffix:
            members = [f"{t}.JK" for t in members]
        result[label] = members
    return result


def sector_options() -> list:
    """List label sektor (emoji + nama) untuk dropdown/multiselect UI, urut kode A-K."""
    return [f"{info['emoji']} {info['name']}" for _, info in sorted(SECTORS.items())]

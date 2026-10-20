# -*- coding: utf-8 -*-
"""
SESI 1 - Analitik dan AI: studi kasus PT MAJU BERSAMA (dataset SINTETIS)

Mengikuti materi Andria Arisal:
  * ~120 pelanggan, 24 produk, ~24.000 baris invoice, 1 Jul 2024 - 30 Jun 2026
  * Kondisi: pendapatan NAIK, laba bersih TURUN, piutang (accounts receivable) NAIK
  * Alur: pemahaman bisnis -> persiapan data -> modeling (descriptive, diagnostic,
    predictive, prescriptive) -> evaluasi -> rekomendasi
  * Pemeriksaan kualitas data: missing, duplikasi, kejadian tak mungkin, kategori tidak
    konsisten, relasi tabel rusak, diskon ekstrem
  * Jebakan granularitas join, class imbalance (accuracy ~89% tetapi recall ~0), data leakage
  * Perbaikan: buang leakage, class_weight, fitur riwayat + kalender, temporal split,
    metrik recall/precision
  * Next steps: operating expenses, waktu bayar, segmen pelanggan, margin per produk

PENTING: datanya dibuat oleh kode ini agar cocok dengan cerita materi. Bukan data asli
PT Maju Bersama milik pemateri. Jalankan: python sesi1_pt_maju_bersama.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(2024)
N_CUST, N_PROD, N_INV, T = 120, 24, 24_000, 24
BULAN = pd.period_range("2024-07", "2026-06", freq="M")


def judul(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ============================================================ 1. DATA SINTETIS
cust = pd.DataFrame({
    "cust_id": np.arange(N_CUST),
    "segmen": rng.choice(["korporat", "UKM", "ritel"], N_CUST, p=[.2, .5, .3]),
    "credit_limit": rng.choice([50, 100, 200, 500], N_CUST) * 1_000_000,
})
base_risk = cust.segmen.map({"korporat": .03, "UKM": .10, "ritel": .07}).values
cust["risk"] = 0.65 * base_risk * rng.lognormal(0, .6, N_CUST)      # sifat pelanggan menetap

prod = pd.DataFrame({
    "prod_id": np.arange(N_PROD),
    "kategori": np.repeat(["A", "B", "C", "D"], N_PROD // 4),
    "unit_cost": rng.uniform(40_000, 400_000, N_PROD),
})
prod["list_price"] = prod.unit_cost * prod.kategori.map({"A": 1.45, "B": 1.40, "C": 1.35, "D": 1.40})

# volume invoice naik sepanjang 24 bulan
bln = rng.choice(T, N_INV, p=np.linspace(1, 1.6, T) / np.linspace(1, 1.6, T).sum())
tgl = pd.to_datetime(BULAN[bln].to_timestamp()) + pd.to_timedelta(rng.integers(0, 28, N_INV), unit="D")
inv = pd.DataFrame({
    "invoice_id": np.arange(N_INV),
    "cust_id": rng.integers(0, N_CUST, N_INV),
    "prod_id": rng.integers(0, N_PROD, N_INV),
    "invoice_date": tgl,
    "qty": rng.integers(1, 30, N_INV),
    "terms_days": rng.choice([14, 30, 45], N_INV),
})
inv = inv.merge(prod[["prod_id", "kategori"]], on="prod_id")
t_idx = inv.invoice_date.dt.to_period("M").map({p: i for i, p in enumerate(BULAN)}).values
# promo kategori C makin agresif -> margin tergerus
inv["discount"] = np.clip(rng.normal(.04, .02, N_INV) + np.where(inv.kategori == "C", .0045 * t_idx, 0), 0, None)
inv["due_date"] = inv.invoice_date + pd.to_timedelta(inv.terms_days, unit="D")

tmp = inv.merge(cust[["cust_id", "risk"]], on="cust_id").merge(prod[["prod_id", "unit_cost", "list_price"]], on="prod_id")
tmp = tmp.sort_values("invoice_id").reset_index(drop=True)
inv = inv.sort_values("invoice_id").reset_index(drop=True)
tm = tmp.invoice_date.dt.to_period("M").map({p: i for i, p in enumerate(BULAN)}).values
p_long = np.clip(tmp.risk.values * (0.7 + 1.2 * tm / (T - 1)), 0, .7)   # risiko telat naik seiring waktu
long_late = rng.random(N_INV) < p_long
inv["days_late"] = np.where(long_late, 31 + rng.gamma(3, 10, N_INV),
                            np.clip(rng.gamma(2, 3, N_INV) - 4, -5, 30)).round()
inv["payment_date"] = inv.due_date + pd.to_timedelta(inv.days_late, unit="D")
inv["unpaid_30d"] = (inv.days_late > 30).astype(int)               # TARGET prediksi
inv["revenue"] = tmp.qty * tmp.list_price * (1 - inv.discount)
inv["cogs"] = tmp.qty * tmp.unit_cost
inv["gross_profit"] = inv.revenue - inv.cogs

# biaya operasional bulanan: tumbuh lebih cepat daripada laba kotor
gp_bln = inv.groupby(inv.invoice_date.dt.to_period("M")).gross_profit.sum()
opex = pd.Series(gp_bln.iloc[0] * 0.70 * (1 + 0.035 * np.arange(T)), index=BULAN, name="opex")

# ---- kotori data (agar pemeriksaan kualitas bermakna) ----
raw = inv.merge(cust[["cust_id", "segmen", "credit_limit"]], on="cust_id")
raw.loc[rng.choice(len(raw), 150, replace=False), "segmen"] = "ukm "           # kategori tak konsisten
raw.loc[raw.segmen == "UKM", "segmen"] = rng.choice(["UKM", "ukm", "Ukm"], (raw.segmen == "UKM").sum(), p=[.97, .02, .01])
raw.loc[rng.choice(len(raw), 120, replace=False), "discount"] = np.nan          # missing
raw.loc[rng.choice(len(raw), 40, replace=False), "discount"] = rng.uniform(.6, .9, 40)  # diskon ekstrem
raw.loc[rng.choice(len(raw), 25, replace=False), "qty"] = 0                     # kejadian tak mungkin
raw.loc[rng.choice(len(raw), 20, replace=False), "cust_id"] = 999               # relasi rusak
raw = pd.concat([raw, raw.sample(60, random_state=1)])                          # duplikasi

# ============================================================ 2. KUALITAS DATA
judul("2. PEMERIKSAAN KUALITAS DATA (persiapan data)")
print(f"baris mentah: {len(raw):,}")
print("missing discount        :", int(raw.discount.isna().sum()))
print("invoice_id duplikat     :", int(raw.invoice_id.duplicated().sum()))
print("qty <= 0 (tak mungkin)  :", int((raw.qty <= 0).sum()))
print("cust_id tidak ada master:", int((~raw.cust_id.isin(cust.cust_id)).sum()))
print("kategori segmen         :", sorted(raw.segmen.dropna().unique()))
print("diskon > 50% (anomali)  :", int((raw.discount > .5).sum()), "-> DIVERIFIKASI MANUSIA, bukan dibuang otomatis")

df = raw.drop_duplicates("invoice_id").copy()
df = df[df.qty > 0]
df = df[df.cust_id.isin(cust.cust_id)]
df["segmen"] = df.segmen.str.strip().str.upper().replace({"UKM": "UKM", "KORPORAT": "korporat", "RITEL": "ritel"})
df["discount"] = df.discount.fillna(df.groupby("kategori").discount.transform("median"))
df["anomali_diskon"] = (df.discount > .5).astype(int)        # ditandai untuk ditinjau pakar domain
df = df.sort_values("invoice_id").reset_index(drop=True)
print(f"baris setelah pembersihan: {len(df):,}")

# ---- granularitas: join yang salah menggandakan baris ----
cust_bulan = pd.DataFrame({"cust_id": np.repeat(cust.cust_id, T), "bulan": np.tile(BULAN, N_CUST)})
salah = df.merge(cust_bulan, on="cust_id")               # level invoice x level pelanggan-bulan
print(f"\n[Granularitas] join benar (m:1): {len(df):,} baris | join salah ke tabel pelanggan-bulan: "
      f"{len(salah):,} baris (terduplikasi ~{len(salah) / len(df):.0f}x -> total pendapatan menjadi salah)")
df["bulan"] = df.invoice_date.dt.to_period("M")

# ============================================================ 3. DESCRIPTIVE
judul("3. DESCRIPTIVE - apa yang terjadi? (juta Rp)")
rev = df.groupby("bulan").revenue.sum()
gp = df.groupby("bulan").gross_profit.sum()
net = gp - opex
ar = pd.Series({b: df[(df.bulan <= b) & (df.payment_date > b.to_timestamp(how="end"))].revenue.sum()
                for b in BULAN})
ringkas = pd.DataFrame({"pendapatan": rev, "laba_bersih": net, "piutang_akhir_bulan": ar}) / 1e6
print(ringkas.iloc[[0, 1, 2, -3, -2, -1]].round(0).to_string())
q1, q2 = ringkas.iloc[:3].mean(), ringkas.iloc[-3:].mean()
print("\nperubahan rata-rata 3 bulan pertama -> 3 bulan terakhir:")
print(((q2 / q1 - 1) * 100).round(1).astype(str).add("%").to_string())

# ============================================================ 4. DIAGNOSTIC
judul("4. DIAGNOSTIC - mengapa? (operating expenses, margin produk, segmen pelanggan)")
awal, akhir = df.bulan <= BULAN[5], df.bulan >= BULAN[-6]
print("Opex vs laba kotor (3 bln awal -> 3 bln akhir, juta Rp):")
print(f"  laba kotor {gp.iloc[:3].mean() / 1e6:,.0f} -> {gp.iloc[-3:].mean() / 1e6:,.0f} | "
      f"opex {opex.iloc[:3].mean() / 1e6:,.0f} -> {opex.iloc[-3:].mean() / 1e6:,.0f}")
mg = pd.DataFrame({"margin_awal_%": df[awal].groupby("kategori").apply(lambda g: g.gross_profit.sum() / g.revenue.sum() * 100),
                   "margin_akhir_%": df[akhir].groupby("kategori").apply(lambda g: g.gross_profit.sum() / g.revenue.sum() * 100)})
print("\nMargin per kategori produk (6 bln awal vs 6 bln akhir):\n", mg.round(1).to_string())
tel = pd.DataFrame({"telat>30h_awal_%": df[awal].groupby("segmen").unpaid_30d.mean() * 100,
                    "telat>30h_akhir_%": df[akhir].groupby("segmen").unpaid_30d.mean() * 100})
print("\nInvoice belum dibayar 30 hari setelah jatuh tempo per segmen:\n", tel.round(1).to_string())

# ============================================================ 5. PREDICTIVE
judul("5. PREDICTIVE - invoice mana yang masih belum dibayar 30 hari setelah jatuh tempo?")
print(f"proporsi kelas positif: {df.unpaid_30d.mean():.1%}  (class imbalance)")


def riwayat_telat(d, prior):
    """Rasio telat historis pelanggan, hanya dari invoice yang hasilnya SUDAH diketahui
    saat invoice ini terbit (hasil diketahui pada jatuh tempo + 30 hari). Tanpa kebocoran."""
    out = pd.Series(index=d.index, dtype=float)
    for _, g in d.groupby("cust_id"):
        known = (g.due_date + pd.Timedelta(days=30)).values
        o = np.argsort(known)
        ks, cs = known[o], np.cumsum(g.unpaid_30d.values[o])
        n = np.searchsorted(ks, g.invoice_date.values, side="right")
        k = np.where(n > 0, cs[np.maximum(n - 1, 0)], 0)
        out.loc[g.index] = (k + prior * 5) / (n + 5)
    return out


df["riwayat_telat"] = riwayat_telat(df, prior=df.unpaid_30d.mean())
df["bulan_ke"] = df.invoice_date.dt.month                            # fitur kalender
df["akhir_tahun"] = df.invoice_date.dt.month.isin([11, 12]).astype(int)
df["promo"] = (df.discount > .08).astype(int)
seg = pd.get_dummies(df.segmen, prefix="seg").astype(float)
dasar = ["qty", "discount", "terms_days", "credit_limit", "revenue"]
X_dasar = pd.concat([df[dasar], seg], axis=1)
X_baik = pd.concat([X_dasar, df[["riwayat_telat", "bulan_ke", "akhir_tahun", "promo"]]], axis=1)
X_leak = X_dasar.assign(days_late=df.days_late)                      # BOCOR: info setelah pembayaran
y = df.unpaid_30d

cut = pd.Timestamp("2025-12-31")          # temporal split: latih masa lalu, uji masa depan
tr, te = (df.invoice_date <= cut).values, (df.invoice_date > cut).values
print(f"latih: {tr.sum():,} invoice (s.d. Des 2025) | uji: {te.sum():,} invoice (Jan-Jun 2026)\n")


def jalankan(nama, model, X):
    model.fit(X[tr], y[tr])
    p = model.predict(X[te])
    print(f"{nama:<50} accuracy={accuracy_score(y[te], p):.3f} recall={recall_score(y[te], p, zero_division=0):.3f} "
          f"precision={precision_score(y[te], p, zero_division=0):.3f} F1={f1_score(y[te], p, zero_division=0):.3f}")
    return model


rf = lambda **k: RandomForestClassifier(200, random_state=0, n_jobs=-1, **k)
jalankan("A. Random Forest, fitur dasar, tanpa penanganan", rf(), X_dasar)
jalankan("A2. Logistic Regression, fitur dasar", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), X_dasar)
jalankan("B. + fitur BOCOR days_late (tidak realistis)", rf(), X_leak)
jalankan("C. RF + class_weight, fitur dasar", rf(class_weight="balanced_subsample", min_samples_leaf=5), X_dasar)
model = jalankan("D. RF + class_weight + riwayat + kalender", rf(class_weight="balanced_subsample", min_samples_leaf=5), X_baik)

print("\nCatatan:")
print(" A  : accuracy tinggi tetapi recall ~0 -> model hampir selalu menebak 'akan dibayar'.")
print(" B  : hampir sempurna karena leakage -> fitur ini harus DIBUANG.")
print(" C/D: recall naik setelah imbalance ditangani dan fitur diperkaya (riwayat pelanggan, kalender, promo).")

# ============================================================ 6. PRESCRIPTIVE
judul("6. PRESCRIPTIVE - apa yang sebaiknya dilakukan?")
uji = df[te].copy()
uji["skor_risiko"] = model.predict_proba(X_baik[te])[:, 1]
per_cust = (uji.groupby(["cust_id", "segmen", "credit_limit"]).agg(skor=("skor_risiko", "mean"),
                                                                    nilai_invoice=("revenue", "sum"))
            .reset_index().sort_values("skor", ascending=False))
print("10 pelanggan berisiko tertinggi (kandidat penagihan terarah / pengetatan credit limit):")
print(per_cust.head(10).assign(nilai_invoice=lambda d: (d.nilai_invoice / 1e6).round(0)).round(2).to_string(index=False))

# simulasi skenario SEDERHANA (asumsi, bukan hasil pengukuran): penagihan dini ke 20% invoice berisiko tertinggi
asumsi = 0.30
k = int(len(uji) * .20)
tinggi = uji.nlargest(k, "skor_risiko")
selamat = (tinggi.unpaid_30d * tinggi.revenue).sum() * asumsi
total = (uji.unpaid_30d * uji.revenue).sum()
print(f"\nSkenario (ASUMSI penagihan dini mengurangi telat {asumsi:.0%} pada 20% invoice teratas):")
print(f"  piutang telat >30 hari: Rp{total / 1e6:,.0f} juta -> sekitar Rp{(total - selamat) / 1e6:,.0f} juta")
print("  (Materi menyebut simulasi Markov/multi-agent untuk skenario yang lebih lengkap.)")
print(f"  Diskon ekstrem yang perlu ditinjau manusia: {int(df.anomali_diskon.sum())} invoice.")

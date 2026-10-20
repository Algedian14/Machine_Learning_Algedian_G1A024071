# -*- coding: utf-8 -*-
"""
SESI 5 - IMK dan Visualisasi Data: dasbor yang dirancang MENGIKUTI isi modul

 contoh kebutuhan konkret di modul adalah:
  * Fungsional  : "penyaringan data berdasarkan tanggal" dan "ekspor grafik ke PDF"
  * Nonfungsional: "kecepatan pemuatan grafik, kemudahan penggunaan, keterbacaan antarmuka"
Karena itu studi kasusnya (dasbor penjualan toko, persona, dan data) dibuat oleh kode ini
sebagai wadah untuk menerapkan konsep modul.

Konsep modul yang diterapkan:
  M5  User Persona + Use Scenario -> turunan kebutuhan fungsional / nonfungsional (diuji otomatis)
  M2  Beban kognitif (extraneous load dikurangi), konsistensi warna (Similarity pada Gestalt)
  M3  Aksesibilitas WCAG (Perceivable): rasio kontras >= 4,5:1
  M4  Prinsip interaksi: Visibility, Feedback, Constraints, Consistency, Affordance
  M6  Pemilihan jenis grafik, Chartjunk vs Data-Ink (Tufte), Overview -> zoom/filter -> detail (Shneiderman)
"""
import os
import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "output_imk"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(5)


def judul(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ============================================================ M5: PERSONA + SKENARIO
@dataclass
class Persona:
    nama: str
    profil: str
    goals: list = field(default_factory=list)
    pain_points: list = field(default_factory=list)
    perilaku: list = field(default_factory=list)


@dataclass
class Skenario:
    konteks: str
    pemicu: str
    tindakan: list
    hasil: str


persona = Persona(
    nama="Rina (fiktif)", profil="Staf admin toko, 28 tahun, terbiasa Excel, memakai laptop dan HP",
    goals=["Melihat tren penjualan dengan cepat", "Melaporkan penjualan bulanan ke pemilik"],
    pain_points=["Grafik lama berantakan dan sulit dibaca", "Menyalin grafik ke laporan memakan waktu"],
    perilaku=["Bekerja di sela melayani pelanggan", "Mengirim laporan dalam bentuk PDF"],
)
skenario = Skenario(
    konteks="Akhir bulan, Rina menyiapkan laporan untuk pemilik toko",
    pemicu="Pemilik meminta ringkasan penjualan Juni",
    tindakan=["Membuka dasbor (overview setahun)", "Memilih rentang tanggal Juni (zoom/filter)",
              "Memilih kategori yang diminati", "Melihat rincian satu hari (detail)", "Mengekspor grafik ke PDF"],
    hasil="Laporan PDF siap dikirim tanpa menyalin manual",
)
judul("M5. PERSONA DAN SKENARIO")
print(f"{persona.nama}: {persona.profil}")
print("Goals       :", "; ".join(persona.goals))
print("Pain points :", "; ".join(persona.pain_points))
print("Skenario    :", skenario.konteks, "->", skenario.pemicu)
for n, t in enumerate(skenario.tindakan, 1):
    print(f"   {n}. {t}")
print("Hasil       :", skenario.hasil)

# ============================================================ DATA SINTETIS
tanggal = pd.date_range("2025-01-01", "2025-12-31", freq="D")
data = pd.DataFrame({"tanggal": np.repeat(tanggal, 3), "kategori": np.tile(["Makanan", "Minuman", "Snack"], len(tanggal))})
data["jumlah"] = (data.kategori.map({"Makanan": 120, "Minuman": 90, "Snack": 60})
                  + np.linspace(0, 30, len(tanggal)).repeat(3) + rng.normal(0, 12, len(data))).round()
data["harga"] = data.kategori.map({"Makanan": 25, "Minuman": 12, "Snack": 8}) + rng.normal(0, 1.5, len(data))

# palet konsisten: satu warna = satu kategori di SEMUA grafik (Similarity/Consistency)
WARNA = {"Makanan": "#1F5FA8", "Minuman": "#C0392B", "Snack": "#2E7D32"}
TEKS, LATAR = "#222222", "#FFFFFF"


# ============================================================ M3: WCAG (Perceivable)
def _lin(c):
    c /= 255
    return c / 12.92 if c <= .03928 else ((c + .055) / 1.055) ** 2.4


def kontras(fg, bg):
    lum = lambda h: sum(w * _lin(int(h[i:i + 2], 16)) for w, i in zip((.2126, .7152, .0722), (1, 3, 5)))
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + .05) / (b + .05)


# ============================================================ DASBOR (M4, M6)
class Dasbor:
    def __init__(self, df):
        self.df = df
        self.status = ""

    def overview(self):
        """Shneiderman 1: overview first (total bulanan)."""
        return self.df.groupby(self.df.tanggal.dt.to_period("M")).jumlah.sum()

    def filter(self, mulai, akhir, kategori=None):
        """Shneiderman 2: zoom and filter. Constraints: rentang tidak valid ditolak.
        Feedback: pesan status menjelaskan hasil aksi."""
        mulai, akhir = pd.Timestamp(mulai), pd.Timestamp(akhir)
        if akhir < mulai:
            raise ValueError("Tanggal akhir tidak boleh sebelum tanggal mulai.")        # Constraints
        m = (self.df.tanggal >= mulai) & (self.df.tanggal <= akhir)
        if kategori:
            m &= self.df.kategori == kategori
        hasil = self.df[m]
        self.status = f"{len(hasil):,} baris ditampilkan ({mulai:%d %b} - {akhir:%d %b %Y})"   # Feedback
        return hasil

    def detail(self, tgl):
        """Shneiderman 3: details-on-demand."""
        d = self.df[self.df.tanggal == pd.Timestamp(tgl)][["tanggal", "kategori", "jumlah"]]
        return d

    def gambar(self, sub, nama_file):
        """Grafik garis (tren waktu) yang bersih: data-ink tinggi, label langsung, judul menyatakan filter (Visibility)."""
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for kat, g in sub.groupby("kategori"):
            ax.plot(g.tanggal, g.jumlah, color=WARNA[kat], lw=1.6)
            ax.text(g.tanggal.iloc[-1], g.jumlah.iloc[-1], f" {kat}", color=WARNA[kat], va="center", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"Penjualan harian | {self.status}", fontsize=11, loc="left", color=TEKS)
        ax.set_ylabel("Unit terjual", fontsize=10)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(nama_file)
        plt.close(fig)


# ============================================================ UJI KEBUTUHAN (M5)
judul("M5. KEBUTUHAN DIUJI OTOMATIS")
d = Dasbor(data)
hasil = []

sub = d.filter("2025-06-01", "2025-06-30")                                   # F1: filter tanggal
ok = sub.tanggal.min() >= pd.Timestamp("2025-06-01") and sub.tanggal.max() <= pd.Timestamp("2025-06-30")
hasil.append(("F1 Filter data berdasarkan tanggal", ok, d.status))

pdf = f"{OUT}/laporan_juni.pdf"                                              # F2: ekspor PDF
d.gambar(sub, pdf)
hasil.append(("F2 Ekspor grafik ke PDF", os.path.exists(pdf) and os.path.getsize(pdf) > 0, pdf))

besar = pd.concat([data] * 20, ignore_index=True)                            # NF1: kecepatan muat (~ 22 ribu x 20 baris)
besar["tanggal"] = pd.to_datetime(besar.tanggal)
t0 = time.perf_counter()
Dasbor(besar).gambar(Dasbor(besar).filter("2025-01-01", "2025-12-31"), f"{OUT}/uji_kecepatan.png")
dt = time.perf_counter() - t0
hasil.append(("NF1 Pemuatan grafik < 2 detik", dt < 2.0, f"{dt:.2f} detik untuk {len(besar):,} baris"))

kontras_min = min(kontras(c, LATAR) for c in list(WARNA.values()) + [TEKS])  # NF3: keterbacaan
hasil.append(("NF3 Keterbacaan: kontras warna >= 4,5:1 (WCAG AA)", kontras_min >= 4.5, f"terendah {kontras_min:.2f}:1"))

try:                                                                          # Constraints (M4)
    d.filter("2025-06-30", "2025-06-01")
    ok_c = False
except ValueError as e:
    ok_c = True
hasil.append(("M4 Constraints: rentang tanggal terbalik ditolak", ok_c, "ValueError ditangkap"))
hasil.append(("M4 Feedback: ada pesan status setelah filter", bool(d.status), d.status))

for nama, ok, ket in hasil:
    print(f"[{'LULUS' if ok else 'GAGAL'}] {nama}  ->  {ket}")
print("\nCatatan: 'kemudahan penggunaan' (NF2) TIDAK bisa diuji otomatis; perlu usability testing")
print("dengan pengguna nyata (mengamati mereka menyelesaikan tugas), bukan hanya kode.")

# ============================================================ M6: OVERVIEW -> FILTER -> DETAIL
judul("M6. SHNEIDERMAN: overview -> zoom/filter -> details-on-demand")
print("Overview (3 bulan pertama):\n", d.overview().head(3).to_string())
sub = d.filter("2025-06-01", "2025-06-30", "Minuman")
print(f"\nZoom/filter: {d.status} | kategori Minuman, total {int(sub.jumlah.sum())}")
print("\nDetail 2025-06-15:\n", d.detail("2025-06-15").to_string(index=False))

# ============================================================ M6: PEMILIHAN GRAFIK + TUFTE
harian = data.groupby("tanggal").jumlah.sum()
total_kat = data.groupby("kategori").jumlah.sum()
fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
ax[0, 0].plot(harian.index, harian.values, color="#1F5FA8", lw=1)
ax[0, 0].set_title("Line chart: tren waktu", fontsize=11)
ax[0, 1].bar(total_kat.index, total_kat.values, color=[WARNA[k] for k in total_kat.index])
ax[0, 1].set_title("Bar chart: perbandingan kategori", fontsize=11)
contoh = data.sample(300, random_state=0)
ax[1, 0].scatter(contoh.harga, contoh.jumlah, s=12, alpha=.6, color="#1F5FA8")
ax[1, 0].set_title("Scatter plot: hubungan harga dan jumlah", fontsize=11)
ax[1, 1].pie(total_kat.values, labels=total_kat.index, autopct="%1.0f%%", wedgeprops=dict(width=.45),
             colors=[WARNA[k] for k in total_kat.index])
ax[1, 1].set_title("Donut chart: proporsi (3 kategori)", fontsize=11)
for a in ax.ravel()[:3]:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/jenis_grafik.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(total_kat.index, total_kat.values, color=["#ff0000", "#00ff00", "#0000ff"], edgecolor="black",
          linewidth=3, hatch="//")
ax[0].grid(True, lw=1.5)
ax[0].set_facecolor("#fff3b0")
ax[0].set_title("Chartjunk: hiasan tidak menambah informasi", fontsize=11)
bars = ax[1].bar(total_kat.index, total_kat.values, color="#1F5FA8")
for b in bars:
    ax[1].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}", ha="center", va="bottom")
for s in ("top", "right", "left"):
    ax[1].spines[s].set_visible(False)
ax[1].set_yticks([])
ax[1].set_title("Data-ink ratio tinggi", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/chartjunk_vs_dataink.png", dpi=120)
plt.close(fig)

print(f"\nFile di '{OUT}/':", sorted(os.listdir(OUT)))

# ============================================================ TAMPILKAN HASIL (Colab / Jupyter)
# Bagian ini ditambahkan agar grafik terlihat langsung di notebook.
# Jika dijalankan sebagai skrip biasa (python file.py), bagian ini dilewati tanpa error.
judul("MENAMPILKAN HASIL")
try:
    notebook = get_ipython() is not None          # noqa: F821  (hanya ada di Colab/Jupyter)
except NameError:
    notebook = False

if notebook:
    from IPython.display import Image, display

    for f in ["jenis_grafik.png", "chartjunk_vs_dataink.png", "uji_kecepatan.png"]:
        print(f"\n{f}")
        display(Image(f"{OUT}/{f}"))

    # PDF tidak bisa ditampilkan di dalam sel, jadi diunduh ke komputer
    try:
        from google.colab import files
        files.download(f"{OUT}/laporan_juni.pdf")
        print("\nlaporan_juni.pdf sedang diunduh oleh browser.")
    except ImportError:
        print("\nBukan Colab: buka laporan_juni.pdf langsung dari folder", OUT)
else:
    print(f"Bukan notebook: semua file ada di folder '{OUT}/'. Buka secara manual.")

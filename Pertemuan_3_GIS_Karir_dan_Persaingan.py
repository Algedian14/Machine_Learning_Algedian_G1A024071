# -*- coding: utf-8 -*-
"""
Sesi 3 - Karir dan Persaingan di Era Industri dengan GIS (DATA SINTETIS)

Mengimplementasikan contoh yang disebut pemateri, hanya dengan numpy/pandas/scikit-learn
(tanpa GeoPandas/QGIS agar mudah dijalankan di Colab):
  1. Pemilihan lokasi gerai minimarket (contoh Indomaret): radius, jarak antargerai, populasi
  2. Pelaporan warga ala "snap and solve": titik laporan -> aset terdekat (spatial join)
  3. Klasifikasi tutupan lahan otomatis dari citra (remote sensing) + luas per kelas
  4. Luas lahan dari koordinat (shoelace pada koordinat meter)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

rng = np.random.default_rng(11)
R_BUMI = 6_371_000.0                          # jari-jari bumi (meter)
LAT0, LON0 = -3.80, 102.26                    # pusat area contoh (sekitar Bengkulu)


def to_rad(lat, lon):
    return np.radians(np.c_[lat, lon])        # BallTree haversine butuh [lat, lon] dalam radian


def judul(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------------
judul("1. PEMILIHAN LOKASI GERAI: radius, jarak antargerai, populasi")
# titik populasi (sel grid ~ 200 m) dan gerai yang sudah ada
n_pop = 600
pop = pd.DataFrame({
    "lat": LAT0 + rng.normal(0, 0.02, n_pop),
    "lon": LON0 + rng.normal(0, 0.02, n_pop),
    "penduduk": rng.integers(20, 300, n_pop),
})
stores = pd.DataFrame({"lat": LAT0 + rng.normal(0, 0.02, 25),
                       "lon": LON0 + rng.normal(0, 0.02, 25)})
cand = pd.DataFrame({"lat": LAT0 + rng.normal(0, 0.02, 200),
                     "lon": LON0 + rng.normal(0, 0.02, 200)})

pop_tree = BallTree(to_rad(pop.lat, pop.lon), metric="haversine")
store_tree = BallTree(to_rad(stores.lat, stores.lon), metric="haversine")

RADIUS_M, MIN_JARAK_M = 1000, 500
# penduduk dalam radius 1 km dari tiap kandidat
idx = pop_tree.query_radius(to_rad(cand.lat, cand.lon), r=RADIUS_M / R_BUMI)
cand["penduduk_1km"] = [int(pop.penduduk.values[i].sum()) for i in idx]
# jarak ke gerai terdekat (agar tidak saling mengganggu)
d, _ = store_tree.query(to_rad(cand.lat, cand.lon), k=1)
cand["jarak_gerai_terdekat_m"] = (d[:, 0] * R_BUMI).round()

layak = cand[cand.jarak_gerai_terdekat_m >= MIN_JARAK_M]
top = layak.sort_values("penduduk_1km", ascending=False).head(3)
print(f"kandidat: {len(cand)}, lolos syarat jarak >= {MIN_JARAK_M} m: {len(layak)}")
print("3 lokasi terbaik (penduduk terbanyak dalam 1 km):")
print(top.round(5).to_string(index=False))

# ---------------------------------------------------------------------
judul("2. SPATIAL JOIN: laporan warga -> aset terdekat (snap and solve)")
assets = pd.DataFrame({
    "aset_id": range(40),
    "jenis": rng.choice(["pohon", "taman_bermain", "perpustakaan", "bermen_rumput"], 40,
                        p=[.5, .15, .1, .25]),            # atribut seragam antar instansi
    "lat": LAT0 + rng.normal(0, 0.01, 40),
    "lon": LON0 + rng.normal(0, 0.01, 40),
})
reports = pd.DataFrame({
    "laporan_id": range(6),
    "lat": LAT0 + rng.normal(0, 0.01, 6),
    "lon": LON0 + rng.normal(0, 0.01, 6),                 # titik dari foto bergeotag
    "keluhan": ["pohon miring", "rumput tinggi", "ayunan rusak",
                "pohon tumbang", "lampu mati", "rumput tinggi"],
})
tree = BallTree(to_rad(assets.lat, assets.lon), metric="haversine")
d, i = tree.query(to_rad(reports.lat, reports.lon), k=1)
joined = reports.assign(aset_id=assets.aset_id.values[i[:, 0]],
                        jenis_aset=assets.jenis.values[i[:, 0]],
                        jarak_m=(d[:, 0] * R_BUMI).round())
print(joined[["laporan_id", "keluhan", "aset_id", "jenis_aset", "jarak_m"]].to_string(index=False))
print("Laporan dengan jarak > 100 m ke aset terdekat perlu diverifikasi (mungkin bukan aset kota).")

# ---------------------------------------------------------------------
judul("3. KLASIFIKASI TUTUPAN LAHAN OTOMATIS (remote sensing)")
# Reflektansi 4 band (biru, hijau, merah, NIR) per kelas -- nilai sintetis untuk ilustrasi
kelas = {
    0: ("air",        [0.08, 0.07, 0.05, 0.03]),
    1: ("hutan",      [0.03, 0.06, 0.03, 0.45]),
    2: ("sawit",      [0.04, 0.08, 0.05, 0.35]),
    3: ("padi",       [0.05, 0.10, 0.07, 0.28]),
    4: ("permukiman", [0.15, 0.17, 0.19, 0.22]),
}


def sampel(n_per_kelas):
    X, y = [], []
    for k, (_, mean) in kelas.items():
        X.append(np.clip(rng.normal(mean, 0.025, (n_per_kelas, 4)), 0.001, 1))
        y += [k] * n_per_kelas
    return np.vstack(X), np.array(y)


def tambah_ndvi(b):                                       # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (b[:, 3] - b[:, 2]) / (b[:, 3] + b[:, 2] + 1e-9)
    return np.c_[b, ndvi]


Xb, y = sampel(400)
X = tambah_ndvi(Xb)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
rf = RandomForestClassifier(150, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
pred = rf.predict(X_te)
print("akurasi uji:", round(accuracy_score(y_te, pred), 3))
print("confusion matrix (baris=aktual, kolom=prediksi) urutan:",
      [v[0] for v in kelas.values()])
print(confusion_matrix(y_te, pred))

# terapkan ke "citra" 100 x 100 piksel (piksel 10 m -> 1 piksel = 0,01 ha)
H = W = 100
peta_asli = np.zeros((H, W), dtype=int)
peta_asli[:, :30] = 2                  # blok sawit
peta_asli[:, 30:55] = 3                # blok padi
peta_asli[:40, 55:] = 1                # hutan
peta_asli[40:70, 55:] = 4              # permukiman
peta_asli[70:, 55:] = 0                # air
band = np.array([np.clip(rng.normal(kelas[k][1], 0.03), 0.001, 1) for k in peta_asli.ravel()])
hasil = rf.predict(tambah_ndvi(band)).reshape(H, W)
luas_ha = pd.Series(hasil.ravel()).map(lambda k: kelas[k][0]).value_counts() * 0.01
print("\nluas per kelas hasil klasifikasi (hektar):")
print(luas_ha.round(1).to_string())
print("akurasi piksel pada citra:", round(float((hasil == peta_asli).mean()), 3))
print("Poligon hasil klasifikasi bisa diekspor (mis. rasterio/GDAL polygonize) -> tanpa digitasi manual.")

# ---------------------------------------------------------------------
judul("4. LUAS PERSIL DARI KOORDINAT (shoelace)")


def luas_m2(lat, lon):
    # proyeksi lokal sederhana lat/lon -> meter (cukup akurat untuk area kecil)
    x = np.radians(lon - lon.mean()) * R_BUMI * np.cos(np.radians(lat.mean()))
    yy = np.radians(lat - lat.mean()) * R_BUMI
    return 0.5 * abs(np.dot(x, np.roll(yy, -1)) - np.dot(yy, np.roll(x, -1)))


# persegi sekitar 100 m x 100 m di sekitar Bengkulu -> ~1 hektar
dlat = 100 / 111_320
dlon = 100 / (111_320 * np.cos(np.radians(LAT0)))
lat = np.array([LAT0, LAT0, LAT0 + dlat, LAT0 + dlat])
lon = np.array([LON0, LON0 + dlon, LON0 + dlon, LON0])
print(f"luas persil contoh: {luas_m2(lat, lon):,.0f} m2 (~{luas_m2(lat, lon) / 10_000:.2f} ha)")
print("Untuk data nyata, gunakan CRS terproyeksi (mis. UTM 47S/EPSG:32747 untuk Bengkulu).")

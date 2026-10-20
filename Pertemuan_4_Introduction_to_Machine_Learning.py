# -*- coding: utf-8 -*-
"""
SESI 4
Introduction to Machine Learning - Implementasi Kode Studi Kasus
Berdasarkan materi webinar "Informatic Practitioner Lecture Series 2026" (Sesi 4)

Studi kasus yang diimplementasikan (sesuai urutan materi):
  1. Workflow ML + Regresi : Prediksi harga rumah
  2. Regresi               : Prediksi nilai ujian mahasiswa
  3. Klasifikasi           : Deteksi spam & persetujuan pinjaman (loan approval)
                             + metrik (accuracy, precision, recall, F1, confusion matrix)
                             + overfitting vs underfitting
  4. Clustering            : Segmentasi pelanggan (K-Means, Hierarchical, DBSCAN, GMM)
  5. Dimensionality Red.   : PCA
  6. Anomaly Detection     : Deteksi transaksi kartu kredit mencurigakan
  7. Association Rule      : Market basket (support, confidence, lift)
  8. Reinforcement Learning: Q-Learning untuk simulasi "self-driving car"
  9. Semi-supervised       : Self-training (sedikit label, banyak data tanpa label)
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    silhouette_score,
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_classification, load_wine
from sklearn.semi_supervised import SelfTrainingClassifier

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
SEED = 42


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# ======================================================================
# 1. WORKFLOW ML + REGRESI: PREDIKSI HARGA RUMAH
#    Workflow: kumpulkan data -> preprocessing (missing value, duplikat,
#    encoding, normalisasi) -> split 80/20 -> latih -> evaluasi -> prediksi
# ======================================================================
def studi_kasus_harga_rumah():
    judul("1. PREDIKSI HARGA RUMAH (Regresi + Workflow ML)")

    # --- (a) Collect data (sintetis) ---
    n = 600
    df = pd.DataFrame({
        "luas_m2": RNG.integers(36, 300, n).astype(float),
        "kamar_tidur": RNG.integers(1, 6, n).astype(float),
        "kamar_mandi": RNG.integers(1, 4, n).astype(float),
        "lokasi": RNG.choice(["Kota", "Pinggiran", "Desa"], n, p=[0.4, 0.4, 0.2]),
    })
    faktor_lokasi = df["lokasi"].map({"Kota": 1.6, "Pinggiran": 1.0, "Desa": 0.6})
    df["harga_juta"] = (
        8 * df["luas_m2"] + 60 * df["kamar_tidur"] + 40 * df["kamar_mandi"]
    ) * faktor_lokasi + RNG.normal(0, 80, n)

    # Sengaja kotori data seperti data dunia nyata
    df.loc[RNG.choice(n, 25, replace=False), "luas_m2"] = np.nan        # missing value
    df.loc[RNG.choice(n, 15, replace=False), "kamar_mandi"] = np.nan
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)  # duplikat
    print(f"Data awal      : {df.shape}  | missing: {int(df.isna().sum().sum())} | duplikat: {int(df.duplicated().sum())}")

    # --- (b) Preprocessing ---
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns="harga_juta")
    y = df["harga_juta"]
    kolom_num = ["luas_m2", "kamar_tidur", "kamar_mandi"]
    kolom_kat = ["lokasi"]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), kolom_num),        # isi missing + normalisasi
        ("kat", OneHotEncoder(handle_unknown="ignore"), kolom_kat),         # teks -> angka
    ])

    # --- (c) Split training 80% / testing 20% ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Split          : train={len(X_train)}  test={len(X_test)}")

    # --- (d) Pilih model, latih, evaluasi ---
    model_dict = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=SEED),
    }
    terbaik, skor_terbaik = None, -np.inf
    for nama, reg in model_dict.items():
        pipe = Pipeline([("prep", preprocess), ("model", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae, r2 = mean_absolute_error(y_test, pred), r2_score(y_test, pred)
        print(f"{nama:<26} MAE={mae:8.2f} juta | R2={r2:.3f}")
        if r2 > skor_terbaik:
            terbaik, skor_terbaik = pipe, r2

    # --- (e) Deploy / inference pada data baru ---
    rumah_baru = pd.DataFrame([{
        "luas_m2": 120, "kamar_tidur": 3, "kamar_mandi": 2, "lokasi": "Kota"
    }])
    print(f"\nPrediksi rumah baru (120 m2, 3 KT, 2 KM, Kota): "
          f"Rp {terbaik.predict(rumah_baru)[0]:,.0f} juta")


# ======================================================================
# 2. REGRESI: PREDIKSI NILAI UJIAN MAHASISWA
#    Fitur: jam belajar, persentase kehadiran, nilai sebelumnya -> nilai akhir
# ======================================================================
def studi_kasus_nilai_mahasiswa():
    judul("2. PREDIKSI NILAI UJIAN MAHASISWA (Supervised - Regresi)")

    n = 300
    jam = RNG.uniform(1, 12, n)
    hadir = RNG.uniform(50, 100, n)
    prev = RNG.uniform(40, 95, n)
    nilai = 0.25 * prev + 0.30 * hadir + 2.2 * jam + RNG.normal(0, 4, n)
    nilai = np.clip(nilai, 0, 100)

    df = pd.DataFrame({"jam_belajar": jam, "kehadiran_persen": hadir,
                       "nilai_sebelumnya": prev, "nilai_akhir": nilai})
    X, y = df.drop(columns="nilai_akhir"), df["nilai_akhir"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"MAE={mean_absolute_error(y_te, pred):.2f} | R2={r2_score(y_te, pred):.3f}")
    print("Koefisien (pengaruh tiap fitur terhadap nilai akhir):")
    for f, c in zip(X.columns, model.coef_):
        print(f"   {f:<20}: {c:+.3f}")

    mhs = pd.DataFrame([{"jam_belajar": 4, "kehadiran_persen": 70, "nilai_sebelumnya": 60}])
    print(f"Prediksi (belajar 4 jam, hadir 70%, nilai lalu 60): {model.predict(mhs)[0]:.1f}")


# ======================================================================
# 3a. KLASIFIKASI: DETEKSI SPAM (spam / not spam)
# ======================================================================
def studi_kasus_spam():
    judul("3a. DETEKSI SPAM EMAIL (Supervised - Klasifikasi)")

    kata_spam = ["menang", "hadiah", "gratis", "klik", "segera", "diskon", "transfer",
                 "pinjaman", "bonus", "hubungi", "promo", "undian"]
    kata_normal = ["rapat", "tugas", "jadwal", "laporan", "dosen", "kuliah", "proyek",
                   "besok", "dokumen", "presentasi", "kelompok", "revisi"]
    filler = ["anda", "kami", "untuk", "hari", "ini", "sekarang", "dengan", "yang", "dan"]

    def buat(daftar, k):
        return " ".join(RNG.choice(daftar, 4).tolist() + RNG.choice(filler, 3).tolist()
                        + (RNG.choice(kata_normal, 1).tolist() if daftar is kata_spam and RNG.random() < 0.3 else []))

    teks = [buat(kata_spam, 1) for _ in range(250)] + [buat(kata_normal, 0) for _ in range(250)]
    label = ["spam"] * 250 + ["bukan_spam"] * 250

    X_tr, X_te, y_tr, y_te = train_test_split(teks, label, test_size=0.2, random_state=SEED, stratify=label)
    model = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Accuracy: {accuracy_score(y_te, pred):.3f}")
    print(classification_report(y_te, pred))

    for email in ["selamat anda menang hadiah gratis klik segera",
                  "jadwal rapat proyek besok dan revisi laporan"]:
        print(f"'{email}'  ->  {model.predict([email])[0]}")


# ======================================================================
# 3b. KLASIFIKASI: PERSETUJUAN PINJAMAN (LOAN APPROVAL) + METRIK EVALUASI
#     + Overfitting / Underfitting
# ======================================================================
def studi_kasus_loan():
    judul("3b. LOAN APPROVAL (Klasifikasi biner) + Evaluasi Model")

    n = 1000
    df = pd.DataFrame({
        "pendapatan_juta": RNG.normal(12, 5, n).clip(2, 50),
        "skor_kredit": RNG.normal(650, 80, n).clip(300, 850),
        "rasio_utang": RNG.uniform(0.05, 0.9, n),
        "status_kerja": RNG.choice(["tetap", "kontrak", "wirausaha", "menganggur"], n, p=[.45, .25, .2, .1]),
    })
    bonus = df["status_kerja"].map({"tetap": 1.0, "kontrak": 0.4, "wirausaha": 0.3, "menganggur": -1.5})
    skor = (0.12 * df["pendapatan_juta"] + 0.015 * (df["skor_kredit"] - 650)
            - 3.0 * df["rasio_utang"] + bonus)
    prob = 1 / (1 + np.exp(-(skor - 0.6)))
    df["disetujui"] = (RNG.random(n) < prob).astype(int)   # 1=approve, 0=reject
    print(f"Distribusi label: {df['disetujui'].value_counts().to_dict()}")

    X, y = df.drop(columns="disetujui"), df["disetujui"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    prep = ColumnTransformer([
        ("num", StandardScaler(), ["pendapatan_juta", "skor_kredit", "rasio_utang"]),
        ("kat", OneHotEncoder(), ["status_kerja"]),
    ])
    algoritma = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree (depth=4)": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    print(f"\n{'Algoritma':<26}{'Acc':>7}{'Prec':>7}{'Recall':>8}{'F1':>7}")
    pipes = {}
    for nama, clf in algoritma.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_tr, y_tr)
        pipes[nama] = pipe
        p = pipe.predict(X_te)
        print(f"{nama:<26}{accuracy_score(y_te, p):7.3f}{precision_score(y_te, p):7.3f}"
              f"{recall_score(y_te, p):8.3f}{f1_score(y_te, p):7.3f}")

    print("\nConfusion Matrix Random Forest  [baris=aktual, kolom=prediksi]")
    print(pd.DataFrame(confusion_matrix(y_te, pipes["Random Forest"].predict(X_te)),
                       index=["aktual_ditolak", "aktual_disetujui"],
                       columns=["pred_ditolak", "pred_disetujui"]))

    pemohon = pd.DataFrame([{"pendapatan_juta": 15, "skor_kredit": 720,
                             "rasio_utang": 0.25, "status_kerja": "tetap"}])
    hasil = pipes["Random Forest"].predict(pemohon)[0]
    print(f"\nPemohon baru -> {'DISETUJUI' if hasil == 1 else 'DITOLAK'}")

    # --- Overfitting vs Underfitting: bandingkan akurasi train vs test ---
    print("\nOverfitting / Underfitting (Decision Tree berdasarkan kedalaman):")
    print(f"{'max_depth':<12}{'Acc Train':>10}{'Acc Test':>10}   Keterangan")
    for d in [1, 3, 5, None]:
        pipe = Pipeline([("prep", prep),
                         ("clf", DecisionTreeClassifier(max_depth=d, random_state=SEED))]).fit(X_tr, y_tr)
        a_tr = accuracy_score(y_tr, pipe.predict(X_tr))
        a_te = accuracy_score(y_te, pipe.predict(X_te))
        ket = "underfitting?" if d == 1 else ("OVERFITTING (train terlalu sempurna)" if a_tr - a_te > 0.12 else "wajar")
        print(f"{str(d):<12}{a_tr:10.3f}{a_te:10.3f}   {ket}")


# ======================================================================
# 4. CLUSTERING: SEGMENTASI PELANGGAN (Unsupervised)
# ======================================================================
def studi_kasus_segmentasi_pelanggan():
    judul("4. SEGMENTASI PELANGGAN (Clustering)")

    # Fitur: frekuensi belanja per bulan, rata-rata nilai belanja (ribu rupiah)
    pusat = [[2, 80], [12, 150], [25, 600], [6, 900]]
    X, _ = make_blobs(n_samples=400, centers=pusat, cluster_std=[1.5, 3, 4, 2.5], random_state=SEED)
    df = pd.DataFrame(X, columns=["frekuensi_belanja", "rata2_belanja_ribu"]).abs()
    Xs = StandardScaler().fit_transform(df)          # normalisasi sebelum hitung jarak

    # Menentukan jumlah K: lihat inertia (elbow) dan silhouette
    print("Mencari K terbaik:")
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        print(f"   K={k}  inertia={km.inertia_:8.1f}  silhouette={silhouette_score(Xs, km.labels_):.3f}")

    # K-Means dengan K=4
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Xs)
    df["segmen_kmeans"] = km.labels_
    print("\nProfil tiap segmen (K-Means, K=4):")
    print(df.groupby("segmen_kmeans").agg(jumlah=("frekuensi_belanja", "size"),
                                          rata_frekuensi=("frekuensi_belanja", "mean"),
                                          rata_belanja=("rata2_belanja_ribu", "mean")).round(1))

    # Algoritma clustering lain yang disebut di materi
    hier = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(Xs)
    db = DBSCAN(eps=0.35, min_samples=8).fit(Xs)
    gmm = GaussianMixture(n_components=4, random_state=SEED).fit(Xs)
    print(f"\nSilhouette Hierarchical : {silhouette_score(Xs, hier.labels_):.3f}")
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"DBSCAN                  : {n_db} cluster, {int((db.labels_ == -1).sum())} titik noise/outlier")
    print(f"Silhouette GMM          : {silhouette_score(Xs, gmm.predict(Xs)):.3f}")


# ======================================================================
# 5. DIMENSIONALITY REDUCTION: PCA
# ======================================================================
def studi_kasus_pca():
    judul("5. PCA (Dimensionality Reduction)")
    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    print(f"Dimensi awal: {X.shape[1]} fitur")

    pca_full = PCA().fit(X)
    kumulatif = np.cumsum(pca_full.explained_variance_ratio_)
    print("Varians kumulatif per komponen:", np.round(kumulatif[:6], 3))

    pca = PCA(n_components=0.90).fit(X)              # pertahankan 90% informasi
    Xr = pca.transform(X)
    print(f"Setelah PCA (90% varians): {Xr.shape[1]} komponen")

    # Apakah informasi penting tetap terjaga? Bandingkan akurasi klasifikasi
    for nama, data_x in [("Semua fitur", X), ("Hasil PCA", Xr)]:
        a_tr, a_te, y_tr, y_te = train_test_split(data_x, data.target, test_size=0.2, random_state=SEED)
        acc = accuracy_score(y_te, LogisticRegression(max_iter=1000).fit(a_tr, y_tr).predict(a_te))
        print(f"   Akurasi ({nama:<11}): {acc:.3f}")


# ======================================================================
# 6. ANOMALY DETECTION: TRANSAKSI KARTU KREDIT MENCURIGAKAN
# ======================================================================
def studi_kasus_anomali():
    judul("6. ANOMALY DETECTION - Fraud Kartu Kredit")

    normal = pd.DataFrame({
        "nominal_ribu": RNG.normal(250, 90, 980).clip(10, None),
        "jam": RNG.normal(14, 4, 980).clip(0, 23),
        "jarak_dari_rumah_km": RNG.exponential(8, 980),
    })
    # transaksi luar negeri, nominal besar, tengah malam
    fraud = pd.DataFrame({
        "nominal_ribu": RNG.normal(4000, 800, 20),
        "jam": RNG.choice([1, 2, 3, 4], 20).astype(float),
        "jarak_dari_rumah_km": RNG.normal(12000, 1500, 20),
    })
    df = pd.concat([normal, fraud], ignore_index=True)
    benar = np.r_[np.zeros(980), np.ones(20)]

    iso = IsolationForest(contamination=0.02, random_state=SEED).fit(df)
    pred = (iso.predict(df) == -1).astype(int)
    print("Isolation Forest")
    print(f"   Terdeteksi anomali : {pred.sum()}")
    print(f"   Fraud tertangkap   : {int(((pred == 1) & (benar == 1)).sum())}/20")
    print(f"   False alarm        : {int(((pred == 1) & (benar == 0)).sum())}")

    # Metode statistik (Z-score) sebagai pembanding
    z = np.abs((df["nominal_ribu"] - df["nominal_ribu"].mean()) / df["nominal_ribu"].std())
    print(f"Z-score (|z|>3) pada nominal: {(z > 3).sum()} transaksi ditandai")


# ======================================================================
# 7. ASSOCIATION RULE LEARNING: MARKET BASKET (support, confidence, lift)
# ======================================================================
def studi_kasus_market_basket():
    judul("7. ASSOCIATION RULE - Market Basket (Supermarket)")

    transaksi = (
        [["roti", "susu"]] * 20 + [["roti", "telur"]] * 18 + [["roti", "susu", "telur"]] * 15
        + [["susu", "kopi"]] * 10 + [["beras", "minyak"]] * 15 + [["roti", "selai"]] * 10
        + [["kopi", "gula"]] * 8 + [["beras", "telur", "minyak"]] * 6
    )
    n = len(transaksi)
    print(f"Jumlah transaksi: {n}")

    items = sorted({i for t in transaksi for i in t})
    sup_item = {i: sum(i in t for t in transaksi) / n for i in items}

    rows = []
    for a in items:
        for b in items:
            if a == b:
                continue
            sup_ab = sum((a in t and b in t) for t in transaksi) / n
            if sup_ab == 0:
                continue
            conf = sup_ab / sup_item[a]
            lift = conf / sup_item[b]
            rows.append({"jika_beli": a, "maka_beli": b, "support": sup_ab,
                         "confidence": conf, "lift": lift})
    aturan = pd.DataFrame(rows)
    aturan = aturan[(aturan.support >= 0.05) & (aturan.confidence >= 0.4)]
    print("\nAturan teratas (urut lift):")
    print(aturan.sort_values("lift", ascending=False).head(8).round(3).to_string(index=False))
    print("\nlift > 1  : kedua barang cenderung dibeli bersamaan (letakkan berdekatan / bundling)")


# ======================================================================
# 8. REINFORCEMENT LEARNING: Q-LEARNING "SELF-DRIVING CAR" SEDERHANA
#    Agent = mobil, Environment = grid jalan, Action = atas/bawah/kiri/kanan
#    Reward: +10 tiba di tujuan, -10 tabrakan, -1 tiap langkah
# ======================================================================
def studi_kasus_q_learning():
    judul("8. REINFORCEMENT LEARNING - Q-Learning (Mobil Otonom Sederhana)")

    UKURAN = 5
    START, GOAL = (0, 0), (4, 4)
    RINTANGAN = {(1, 1), (2, 3), (3, 1), (3, 3)}   # kendaraan lain / pejalan kaki
    AKSI = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    NAMA = {0: "atas", 1: "bawah", 2: "kiri", 3: "kanan"}

    def langkah(state, a):
        r, c = state[0] + AKSI[a][0], state[1] + AKSI[a][1]
        if not (0 <= r < UKURAN and 0 <= c < UKURAN):
            return state, -2, False          # menabrak tepi jalan
        if (r, c) in RINTANGAN:
            return (r, c), -10, True         # tabrakan = penalty, episode berakhir
        if (r, c) == GOAL:
            return (r, c), 10, True          # sampai tujuan = reward
        return (r, c), -1, False             # penalti kecil agar rute efisien

    Q = np.zeros((UKURAN, UKURAN, 4))
    alpha, gamma, eps = 0.1, 0.95, 1.0

    for ep in range(2000):
        s, selesai = START, False
        for _ in range(100):
            a = RNG.integers(4) if RNG.random() < eps else int(np.argmax(Q[s]))
            s2, reward, selesai = langkah(s, a)
            Q[s][a] += alpha * (reward + gamma * np.max(Q[s2]) * (not selesai) - Q[s][a])
            s = s2
            if selesai:
                break
        eps = max(0.05, eps * 0.998)          # makin lama makin mengandalkan pengalaman

    # Jalankan policy hasil belajar
    s, rute, aksi_log = START, [START], []
    for _ in range(20):
        a = int(np.argmax(Q[s]))
        s, _, selesai = langkah(s, a)
        rute.append(s)
        aksi_log.append(NAMA[a])
        if selesai:
            break

    print("Peta (S=start, G=goal, X=rintangan, *=rute hasil belajar):")
    for r in range(UKURAN):
        baris = ""
        for c in range(UKURAN):
            if (r, c) == START: baris += " S "
            elif (r, c) == GOAL: baris += " G "
            elif (r, c) in RINTANGAN: baris += " X "
            elif (r, c) in rute: baris += " * "
            else: baris += " . "
        print(baris)
    print(f"\nRute: {' -> '.join(aksi_log)}")
    print("Berhasil sampai tujuan tanpa tabrakan" if rute[-1] == GOAL else "Belum berhasil, tambah episode")


# ======================================================================
# 9. SEMI-SUPERVISED: SELF-TRAINING (sedikit label, banyak data tanpa label)
#    Contoh materi: 1.000 email berlabel vs 30.000 tanpa label
# ======================================================================
def studi_kasus_semi_supervised():
    judul("9. SEMI-SUPERVISED LEARNING - Self-Training")

    X, y = make_classification(n_samples=6000, n_features=20, n_informative=8,
                               class_sep=1.0, random_state=SEED)
    X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    n_label = 100                                     # hanya 100 data berlabel
    y_semi = np.full_like(y_pool, -1)                 # -1 = tidak berlabel
    idx = RNG.choice(len(y_pool), n_label, replace=False)
    y_semi[idx] = y_pool[idx]
    print(f"Data berlabel: {n_label} | Tanpa label: {len(y_pool) - n_label}")

    base = RandomForestClassifier(n_estimators=100, random_state=SEED)
    sup_only = base.fit(X_pool[idx], y_pool[idx])
    acc_sup = accuracy_score(y_te, sup_only.predict(X_te))

    self_train = SelfTrainingClassifier(RandomForestClassifier(n_estimators=100, random_state=SEED),
                                        threshold=0.8).fit(X_pool, y_semi)
    acc_semi = accuracy_score(y_te, self_train.predict(X_te))

    full = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_pool, y_pool)
    acc_full = accuracy_score(y_te, full.predict(X_te))

    print(f"Supervised (100 label saja)   : {acc_sup:.3f}")
    print(f"Semi-supervised (self-training): {acc_semi:.3f}")
    print(f"Supervised (semua label, batas atas): {acc_full:.3f}")


# ======================================================================
if __name__ == "__main__":
    studi_kasus_harga_rumah()
    studi_kasus_nilai_mahasiswa()
    studi_kasus_spam()
    studi_kasus_loan()
    studi_kasus_segmentasi_pelanggan()
    studi_kasus_pca()
    studi_kasus_anomali()
    studi_kasus_market_basket()
    studi_kasus_q_learning()
    studi_kasus_semi_supervised()
    print("\nSelesai. Semua studi kasus berhasil dijalankan.")

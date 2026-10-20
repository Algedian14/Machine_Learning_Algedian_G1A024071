# -*- coding: utf-8 -*-
"""
SESI 6 - Pemantauan Deforestasi Menggunakan Citra Satelit dan Artificial Intelligence

Konsep resume -> bagian kode:
  1. Citra optik (Sentinel-2) terganggu awan, radar (Sentinel-1) tidak       -> Bagian 1
  2. Deforestation alerts dari perubahan tutupan                             -> Bagian 2
  3. Klasifikasi penyebab (pertanian, tambang, tebang, jalan, api, alam)
     memakai CNN yang dilatih dari citra berlabel                            -> Bagian 3
  4. Evaluasi: overall accuracy vs macro F1 (kelas dominan = pertanian),
     performa berbeda antarwilayah                                           -> Bagian 4 dan 5
  5. Validasi lapangan untuk hasil yang tidak yakin                          -> Bagian 6
  6. Frekuensi pembaruan: mingguan/bulanan vs sensor termal untuk kebakaran  -> Bagian 7
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, label
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    ADA_TORCH = True
except ImportError:
    ADA_TORCH = False

warnings.filterwarnings("ignore")
SEED = 6
OUT = "output_deforestasi"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)


def judul(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ====================================================================== KONSTANTA
UK = 16                      # ukuran ubin 16x16 piksel; 1 piksel Sentinel-2 = 10 m -> 0,01 ha
HA_PIKSEL = 0.01
KELAS = ["Pertanian", "Pertambangan", "Penebangan", "Jalan", "Kebakaran", "Gangguan alami"]
SINGKAT = ["Tani", "Tambang", "Tebang", "Jalan", "Api", "Alam"]
# palet Okabe-Ito (aman untuk buta warna); satu warna = satu kelas di SEMUA gambar
WARNA = ["#E69F00", "#444444", "#0072B2", "#CC79A7", "#D55E00", "#009E73"]
SALURAN = ["NDVI sebelum", "NDVI sesudah", "Merah (B4)", "SWIR (B11)", "Radar dVV (dB)", "Radar dVH (dB)"]

# Ciri spektral tiap penyebab: (rata-rata, simpangan antar-sampel). Angkanya ASUMSI ilustratif.
SPEKTRAL = {
    0: dict(ndvi=(0.35, 0.12), red=(0.12, 0.04), swir=(0.20, 0.05), vv=(-1.5, 1.4), vh=(-1.5, 1.4), noise=0.04),
    1: dict(ndvi=(0.16, 0.09), red=(0.22, 0.06), swir=(0.30, 0.06), vv=(0.5, 1.5), vh=(0.2, 1.5), noise=0.05),
    2: dict(ndvi=(0.45, 0.12), red=(0.10, 0.04), swir=(0.18, 0.05), vv=(-0.8, 1.0), vh=(-0.7, 1.0), noise=0.05),
    3: dict(ndvi=(0.28, 0.12), red=(0.17, 0.05), swir=(0.25, 0.06), vv=(-1.0, 1.3), vh=(-1.0, 1.3), noise=0.05),
    4: dict(ndvi=(0.20, 0.09), red=(0.16, 0.05), swir=(0.31, 0.06), vv=(-0.5, 1.0), vh=(-0.5, 1.0), noise=0.05),
    5: dict(ndvi=(0.38, 0.13), red=(0.11, 0.04), swir=(0.17, 0.05), vv=(-2.2, 1.5), vh=(-2.5, 1.5), noise=0.05),
}
# Wilayah B = kondisi berbeda (musim/sensor/vegetasi) -> menguji generalisasi antarwilayah
WILAYAH = {"A": dict(ndvi=0.0, red=0.0, swir=0.0, radar=0.0, noise=1.0),
           "B": dict(ndvi=-0.07, red=0.03, swir=0.04, radar=-0.8, noise=1.4)}


# ====================================================================== GENERATOR DATA
def _blob(r, sigma, frac):
    z = gaussian_filter(r.normal(size=(UK, UK)), sigma, mode="wrap")
    return z > np.quantile(z, 1 - frac)


def _bentuk(kls, r):
    """Bentuk spasial khas tiap penyebab (inilah yang dipelajari 'mata' CNN)."""
    yy, xx = np.mgrid[0:UK, 0:UK]
    if kls == 0:                                   # pertanian: blok persegi panjang
        h, w = r.integers(5, 15, 2)
        y0, x0 = r.integers(0, UK - h + 1), r.integers(0, UK - w + 1)
        return (yy >= y0) & (yy < y0 + h) & (xx >= x0) & (xx < x0 + w)
    if kls == 1:
        return _blob(r, 3, r.uniform(.30, .50))    # tambang: area terbuka tak beraturan
    if kls == 2:                                   # penebangan selektif: bintik kecil tersebar
        m = np.zeros((UK, UK), bool)
        for _ in range(r.integers(5, 10)):
            y, x = r.integers(1, UK - 2, 2)
            m[y:y + r.integers(2, 4), x:x + r.integers(2, 4)] = True
        return m
    if kls == 3:                                   # jalan: garis tipis
        ang = r.uniform(0, np.pi)
        cy, cx = r.uniform(5, 11, 2)
        return np.abs((xx - cx) * np.sin(ang) - (yy - cy) * np.cos(ang)) < r.uniform(.8, 1.5)
    if kls == 4:
        return _blob(r, 3, r.uniform(.35, .60))    # kebakaran: bekas bakar tak beraturan
    return _blob(r, 1.5, r.uniform(.15, .35))      # gangguan alami: bercak-bercak


def buat_patch(kls, r, wil="A"):
    """Ubin 6 saluran x 16 x 16. kls=None -> hutan utuh (tanpa perubahan)."""
    w = WILAYAH[wil]
    nz = w["noise"]
    ndvi_b = 0.80 + r.normal(0, .03, (UK, UK)) * nz
    ndvi_a = 0.80 + w["ndvi"] + r.normal(0, .03, (UK, UK)) * nz
    red = 0.04 + w["red"] + r.normal(0, .01, (UK, UK)) * nz
    swir = 0.12 + w["swir"] + r.normal(0, .01, (UK, UK)) * nz
    vv = r.normal(0, .6, (UK, UK)) * nz
    vh = r.normal(0, .6, (UK, UK)) * nz
    if kls is not None:
        p, m = SPEKTRAL[kls], _bentuk(kls, r)
        n, tek = int(m.sum()), SPEKTRAL[kls]["noise"] * nz
        ndvi_a[m] = r.normal(*p["ndvi"]) + w["ndvi"] + r.normal(0, tek, n)
        red[m] = r.normal(*p["red"]) + w["red"] + r.normal(0, tek / 2, n)
        swir[m] = r.normal(*p["swir"]) + w["swir"] + r.normal(0, tek / 2, n)
        vv[m] = r.normal(*p["vv"]) + w["radar"] + r.normal(0, .6 * nz, n)
        vh[m] = r.normal(*p["vh"]) + w["radar"] + r.normal(0, .6 * nz, n)
    return np.stack([ndvi_b, ndvi_a, red, swir, vv, vh]).astype(np.float32)


def buat_dataset(jumlah, wil, r):
    X, y = [], []
    for k, n in enumerate(jumlah):
        for _ in range(n):
            X.append(buat_patch(k, r, wil))
            y.append(k)
    return np.stack(X), np.array(y)


def buat_scene(r, n_ubin=6, n_event=14, p_kelas=(.30, .10, .20, .15, .10, .15)):
    """Satu 'scene' 96x96 piksel berisi ubin hutan dan ubin gangguan + tutupan awan pada citra optik."""
    N = n_ubin * UK
    scene = np.zeros((6, N, N), np.float32)
    benar = -np.ones((n_ubin, n_ubin), int)
    event = set(r.choice(n_ubin * n_ubin, n_event, replace=False).tolist())
    for i in range(n_ubin):
        for j in range(n_ubin):
            if i * n_ubin + j in event:
                benar[i, j] = r.choice(6, p=p_kelas)
            scene[:, i * UK:(i + 1) * UK, j * UK:(j + 1) * UK] = buat_patch(
                benar[i, j] if benar[i, j] >= 0 else None, r)
    z = gaussian_filter(r.normal(size=(N, N)), 9)
    awan = z > np.quantile(z, .78)                          # ~22% tertutup awan
    return scene, benar, awan


# ====================================================================== MODEL
def fitur_manual(X):
    """Pengganti CNN (jika PyTorch tidak ada): statistik tiap saluran + ciri bentuk area berubah."""
    out = []
    for p in X:
        ubah = (p[0] - p[1]) > 0.25
        f = []
        for c in range(6):
            f += [p[c].mean(), p[c].std(), p[c].min(), p[c].max()]
        n = int(ubah.sum())
        f.append(n / ubah.size)
        if n > 2:
            yy, xx = np.nonzero(ubah)
            ev = np.linalg.eigvalsh(np.cov(np.vstack([yy, xx])))
            f.append(ev[1] / (ev[0] + 1e-3))              # kelonjongan (jalan = sangat lonjong)
        else:
            f.append(1.0)
        lab, nl = label(ubah)
        f += [nl, (np.bincount(lab.ravel())[1:].max() / n) if n > 0 else 0]
        out.append(f)
    return np.array(out)


class ModelFitur:
    nama = "Random Forest (pengganti CNN)"

    def __init__(self, seimbang):
        self.rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                         class_weight="balanced_subsample" if seimbang else None)

    def fit(self, X, y):
        self.rf.fit(fitur_manual(X), y)
        return self

    def predict_proba(self, X):
        return self.rf.predict_proba(fitur_manual(X))


class ModelCNN:
    nama = "CNN (PyTorch)"

    def __init__(self, seimbang, epoch=25):
        self.seimbang, self.epoch = seimbang, epoch

    def fit(self, X, y):
        torch.manual_seed(SEED)
        self.mu = X.mean(axis=(0, 2, 3), keepdims=True)
        self.sd = X.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        self.net = nn.Sequential(
            nn.Conv2d(X.shape[1], 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, len(KELAS)))
        bobot = None
        if self.seimbang:                                    # kelas langka diberi bobot lebih besar
            cnt = np.bincount(y, minlength=len(KELAS))
            bobot = torch.tensor(len(y) / (len(KELAS) * np.maximum(cnt, 1)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=bobot)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        for _ in range(self.epoch):
            self.net.train()
            urut = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 64):
                idx = urut[i:i + 64]
                opt.zero_grad()
                loss_fn(self.net(Xt[idx]), yt[idx]).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            logit = self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32))
            return torch.softmax(logit, dim=1).numpy()


def buat_model(seimbang):
    return ModelCNN(seimbang) if ADA_TORCH else ModelFitur(seimbang)


def ringkas(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0)


# ====================================================================== SCENE (dipakai Bagian 1, 2, 6)
scene_r = np.random.default_rng(SEED + 1)
scene, benar_ubin, awan = buat_scene(scene_r)
N = scene.shape[1]
scene_opt = scene.copy()
scene_opt[1:4, awan] = np.nan             # awan menutup citra OPTIK sesudah (NDVI sesudah, merah, SWIR)

# ====================================================================== 1. OPTIK vs RADAR
judul("1. CITRA OPTIK (SENTINEL-2) vs RADAR (SENTINEL-1)")
print(f"Piksel optik tertutup awan di scene ini : {awan.mean():.0%}")
print("Piksel radar tertutup awan              : 0%  (radar menembus awan)")
p_awan = 0.6      # ASUMSI: peluang sebuah akuisisi tertutup awan di wilayah tropis
print(f"\nAsumsi peluang awan per akuisisi = {p_awan:.0%}, revisit optik 5 hari:")
for hari in (10, 30):
    n = hari // 5
    print(f"   Peluang >=1 citra optik jernih dalam {hari:2d} hari ({n} akuisisi): {1 - p_awan ** n:.0%}")
print("\nKeterbatasan radar: kemampuan mendeteksi objek DI BAWAH kanopi terbatas. Pada simulasi ini")
print("penebangan selektif sengaja diberi perubahan radar yang lemah (lihat SPEKTRAL, kelas 2).")

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
tampil = np.ma.masked_invalid(scene_opt[1])
cm = plt.get_cmap("Greens").copy()
cm.set_bad("#FFFFFF")
im0 = ax[0].imshow(tampil, cmap=cm, vmin=0, vmax=0.9)
ax[0].set_title("Optik: NDVI sesudah\n(putih = tertutup awan)", fontsize=10)
im1 = ax[1].imshow(scene[4], cmap="RdBu", vmin=-4, vmax=4)
ax[1].set_title("Radar: perubahan VV (dB)\n(tidak terganggu awan)", fontsize=10)
ax[2].imshow(awan, cmap="Greys", vmin=0, vmax=1.6)
ax[2].set_title("Mask awan", fontsize=10)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.colorbar(im0, ax=ax[0], fraction=.046); fig.colorbar(im1, ax=ax[1], fraction=.046)
fig.tight_layout(); fig.savefig(f"{OUT}/1_optik_vs_radar.png", dpi=120); plt.close(fig)

# ====================================================================== 2. DEFORESTATION ALERTS
judul("2. DEFORESTATION ALERTS (deteksi perubahan per ubin 16x16 piksel = 2,56 ha)")
AMBANG_DROP, MIN_FRAC = 0.25, 0.05
baris, nb = [], benar_ubin.shape[0]
for i in range(nb):
    for j in range(nb):
        sl = (slice(None), slice(i * UK, (i + 1) * UK), slice(j * UK, (j + 1) * UK))
        t = scene_opt[sl]
        frac_awan = np.isnan(t[1]).mean()
        if frac_awan > 0.5:
            # radar tidak terganggu awan: alert jika >=10% piksel turun lebih dari 1,5 dB
            status = "ALERT RADAR (tertutup awan)" if (t[4] < -1.5).mean() >= 0.10 else "Tertutup awan, tak ada alert"
            frac = np.nan
        else:
            drop = t[0] - t[1]
            frac = float(np.nansum(drop > AMBANG_DROP) / drop.size)
            status = "ALERT" if frac >= MIN_FRAC else "Normal"
        baris.append(dict(i=i, j=j, awan=frac_awan, frac_ubah=frac, luas_ha=(frac * UK * UK * HA_PIKSEL)
                          if frac == frac else np.nan, status=status, benar=benar_ubin[i, j]))
alert = pd.DataFrame(baris)
n_event = int((benar_ubin >= 0).sum())
ada = alert[alert.benar >= 0]
print(f"Ubin berisi gangguan sebenarnya        : {n_event} dari {nb * nb}")
print(f"  terdeteksi ALERT optik               : {int((ada.status == 'ALERT').sum())}")
print(f"  terdeteksi ALERT radar (awan)        : {int(ada.status.str.startswith('ALERT RADAR').sum())}")
print(f"  tertutup awan, tidak terdeteksi      : {int(ada.status.str.startswith('Tertutup').sum())}")
lolos = ada[(ada.status == "Normal")]
print(f"  lolos (ubah < {MIN_FRAC:.0%} piksel)             : {len(lolos)}"
      + (f"  -> {[KELAS[int(k)] for k in lolos.benar]}" if len(lolos) else ""))
print(f"Alarm palsu (ubin hutan utuh jadi ALERT): "
      f"{int(((alert.benar < 0) & alert.status.str.startswith('ALERT')).sum())}")

# ====================================================================== 3. DATASET + CNN
judul("3. KLASIFIKASI PENYEBAB: dataset berlabel dan pelatihan model")
JUMLAH_A = [1200, 120, 500, 250, 100, 200]            # PERTANIAN SANGAT DOMINAN (sesuai resume)
JUMLAH_B = [400, 60, 150, 100, 60, 80]
rA, rB = np.random.default_rng(SEED + 2), np.random.default_rng(SEED + 3)
XA, yA = buat_dataset(JUMLAH_A, "A", rA)
XB, yB = buat_dataset(JUMLAH_B, "B", rB)
Xtr, Xte, ytr, yte = train_test_split(XA, yA, test_size=0.25, random_state=SEED, stratify=yA)
print(f"Model            : {(ModelCNN if ADA_TORCH else ModelFitur).nama}")
print(f"Wilayah A (latih): {len(ytr)} ubin | uji: {len(yte)} ubin | Wilayah B (uji lintas wilayah): {len(yB)} ubin")
print("Distribusi kelas wilayah A:", dict(zip(KELAS, np.bincount(yA).tolist())))

model_biasa = buat_model(seimbang=False).fit(Xtr, ytr)
model_bobot = buat_model(seimbang=True).fit(Xtr, ytr)

# contoh ubin tiap kelas
fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
for k in range(6):
    p = buat_patch(k, np.random.default_rng(k + 50))
    ax[0, k].imshow(p[1], cmap="Greens", vmin=0, vmax=.9)
    ax[0, k].set_title(KELAS[k], fontsize=9, color=WARNA[k])
    ax[1, k].imshow(p[4], cmap="RdBu", vmin=-4, vmax=4)
    ax[0, k].set_xticks([]); ax[0, k].set_yticks([]); ax[1, k].set_xticks([]); ax[1, k].set_yticks([])
ax[0, 0].set_ylabel("NDVI sesudah", fontsize=9); ax[1, 0].set_ylabel("Radar dVV", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/2_contoh_ubin.png", dpi=120); plt.close(fig)

# ====================================================================== 4. EVALUASI
judul("4. EVALUASI: overall accuracy BELUM cukup, lihat macro F1 dan tiap kelas")
hasil = {}
for nama, mdl in [("Tanpa penanganan imbalance", model_biasa), ("Dengan bobot kelas", model_bobot)]:
    pred = mdl.predict_proba(Xte).argmax(1)
    acc, f1m = ringkas(yte, pred)
    hasil[nama] = (pred, acc, f1m, f1_score(yte, pred, average=None, zero_division=0))
    print(f"{nama:<28} overall accuracy={acc:.3f} | macro F1={f1m:.3f}")
print("\nRincian per kelas (model dengan bobot kelas):")
print(classification_report(yte, hasil["Dengan bobot kelas"][0], target_names=KELAS, digits=3, zero_division=0))
print("Catatan: jika overall accuracy tinggi tetapi F1 kelas langka (tambang, kebakaran) rendah,")
print("model tampak bagus padahal sebenarnya lemah pada kelas yang justru penting.")

cmx = confusion_matrix(yte, hasil["Dengan bobot kelas"][0], labels=range(6))
cmn = cmx / cmx.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for a in range(6):
    for b in range(6):
        ax.text(b, a, cmx[a, b], ha="center", va="center", color="white" if cmn[a, b] > .5 else "#222222")
ax.set_xticks(range(6)); ax.set_xticklabels(SINGKAT, fontsize=9)
ax.set_yticks(range(6)); ax.set_yticklabels(KELAS, fontsize=9)
ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya"); ax.set_title("Confusion matrix (wilayah A, data uji)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/3_confusion_matrix.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(6)
ax.bar(x - .2, hasil["Tanpa penanganan imbalance"][3], .4, color="#888888", label="Tanpa penanganan imbalance")
ax.bar(x + .2, hasil["Dengan bobot kelas"][3], .4, color="#0072B2", label="Dengan bobot kelas")
ax.axhline(hasil["Dengan bobot kelas"][1], color="#D55E00", ls="--", lw=1.4)
ax.text(5.5, hasil["Dengan bobot kelas"][1] + .01, "overall accuracy", color="#D55E00", ha="right", fontsize=9)
ax.axhline(hasil["Dengan bobot kelas"][2], color="#222222", ls=":", lw=1.4)
ax.text(5.5, hasil["Dengan bobot kelas"][2] - .05, "macro F1", color="#222222", ha="right", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(KELAS, fontsize=9); ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
ax.set_title("F1 per kelas vs overall accuracy", fontsize=11, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/4_f1_per_kelas.png", dpi=120); plt.close(fig)

# ====================================================================== 5. ANTARWILAYAH
judul("5. PERFORMA BERBEDA ANTARWILAYAH (model dilatih di A, diuji di A dan B)")
pA = model_bobot.predict_proba(Xte).argmax(1)
pB = model_bobot.predict_proba(XB).argmax(1)
aA, fA = ringkas(yte, pA)
aB, fB = ringkas(yB, pB)
print(f"{'Wilayah':<26}{'Accuracy':>10}{'Macro F1':>10}")
print(f"{'A (wilayah latih)':<26}{aA:>10.3f}{fA:>10.3f}")
print(f"{'B (wilayah baru)':<26}{aB:>10.3f}{fB:>10.3f}")
print(f"Penurunan macro F1 dari A ke B: {fA - fB:.3f}")
print("Penyebab di simulasi ini: nilai spektral wilayah B bergeser (musim/sensor/vegetasi berbeda).")
print("Implikasi: model perlu dievaluasi, bahkan dilatih ulang/disesuaikan, untuk tiap wilayah baru.")

# ====================================================================== 6. END-TO-END + VALIDASI LAPANGAN
judul("6. ALERT -> KLASIFIKASI PENYEBAB -> VALIDASI LAPANGAN")
AMBANG_YAKIN = 0.70


def ubin_aman(i, j):
    """Ambil ubin dari scene optik; piksel awan (NaN) diisi nilai hutan agar bisa masuk model."""
    t = scene_opt[:, i * UK:(i + 1) * UK, j * UK:(j + 1) * UK].copy()
    nanm = np.isnan(t[1])
    t[1][nanm], t[2][nanm], t[3][nanm] = t[0][nanm], 0.04, 0.12
    return t


al = alert[alert.status == "ALERT"].copy().reset_index(drop=True)
if len(al):
    proba = model_bobot.predict_proba(np.stack([ubin_aman(r.i, r.j) for r in al.itertuples()]))
    al["prediksi"] = [KELAS[k] for k in proba.argmax(1)]
    al["yakin"] = proba.max(1)
    al["sebenarnya"] = [KELAS[int(k)] if k >= 0 else "(hutan utuh)" for k in al.benar]
    al["benar?"] = np.where(al.prediksi == al.sebenarnya, "ya", "TIDAK")
    al["tindak lanjut"] = np.where(al.yakin < AMBANG_YAKIN, "VERIFIKASI LAPANGAN", "sampel validasi")
    tabel = al[["i", "j", "luas_ha", "prediksi", "yakin", "sebenarnya", "benar?", "tindak lanjut"]].copy()
    tabel["luas_ha"] = tabel.luas_ha.round(2)
    tabel["yakin"] = tabel.yakin.round(2)
    print(tabel.to_string(index=False))
    print(f"\nAkurasi pada {len(al)} ubin alert: {(al['benar?'] == 'ya').mean():.0%} "
          "(sampel kecil, hanya ilustrasi)")
else:
    print("Tidak ada alert optik pada scene ini.")

print("\nTrade-off ambang keyakinan (data uji wilayah A):")
pr = model_bobot.predict_proba(Xte)
print(f"{'Ambang':>8}{'Ubin diterima otomatis':>26}{'Akurasi di sana':>18}")
for th in (0.5, 0.7, 0.9):
    m = pr.max(1) >= th
    print(f"{th:>8.1f}{m.mean():>26.0%}{(pr.argmax(1)[m] == yte[m]).mean():>18.1%}")
print("Makin tinggi ambang -> makin akurat, tetapi makin banyak ubin harus dicek lapangan.")
print("Ingat: citra tidak menjelaskan penyebab mendasar secara lengkap; validasi lapangan tetap perlu.")

# peta alert
fig, ax = plt.subplots(figsize=(7.4, 7.4))
cm2 = plt.get_cmap("Greens").copy()
cm2.set_bad("#FFFFFF")
ax.imshow(np.ma.masked_invalid(scene_opt[1]), cmap=cm2, vmin=0, vmax=.9)
for k in range(1, nb):
    ax.axhline(k * UK - .5, color="#BBBBBB", lw=.5); ax.axvline(k * UK - .5, color="#BBBBBB", lw=.5)
for r in alert.itertuples():
    if r.status.startswith("ALERT RADAR"):
        ax.add_patch(Rectangle((r.j * UK - .5, r.i * UK - .5), UK, UK, fill=False, ec="#222222", lw=2, ls="--"))
        ax.text(r.j * UK + UK / 2 - .5, r.i * UK + UK / 2 - .5, "radar", ha="center", va="center", fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=.8))
for r in al.itertuples():
    k = KELAS.index(r.prediksi)
    ax.add_patch(Rectangle((r.j * UK - .5, r.i * UK - .5), UK, UK, fill=False, ec=WARNA[k], lw=3))
    salah = "" if r.prediksi == r.sebenarnya else " ✗"
    ax.text(r.j * UK + UK / 2 - .5, r.i * UK + UK / 2 - .5, f"{SINGKAT[k]}\n{r.yakin:.0%}{salah}",
            ha="center", va="center", fontsize=8, color="white",
            bbox=dict(fc=WARNA[k], ec="none", alpha=.9, boxstyle="round,pad=.2"))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Peta alert: warna = penyebab prediksi, % = keyakinan, ✗ = salah\n"
             "(putih = awan; garis putus = alert radar saat optik tertutup)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/5_peta_alert.png", dpi=120); plt.close(fig)

# ====================================================================== 7. FREKUENSI PEMBARUAN (KEBAKARAN)
judul("7. FREKUENSI PEMBARUAN: mingguan / bulanan vs sensor termal (kebakaran)")
# ASUMSI ILUSTRATIF: kebakaran mulai kecil lalu meluas ~25% per hari (tidak berasal dari resume)
A0, TUMBUH, TUNDA_PROSES, LUAS_MAKS = 2.0, 1.25, 1.0, 1000.0
r7 = np.random.default_rng(SEED + 7)
opsi = [("Klasifikasi bulanan", 30), ("Alert mingguan", 7), ("Sensor termal (harian)", 1)]
print(f"Asumsi: api awal {A0:.0f} ha, meluas {TUMBUH - 1:.0%}/hari (maks {LUAS_MAKS:,.0f} ha), pemrosesan {TUNDA_PROSES:.0f} hari.")
print(f"{'Sumber data':<26}{'Rata2 jeda':>12}{'Jeda terburuk':>15}{'Luas rata2 saat terdeteksi':>28}")
for nama, T in opsi:
    mulai = r7.uniform(0, 365, 20000)
    jeda = (T - (mulai % T)) % T + TUNDA_PROSES
    luas = np.minimum(A0 * TUMBUH ** jeda, LUAS_MAKS)
    print(f"{nama:<26}{jeda.mean():>9.1f} hr{T + TUNDA_PROSES:>12.0f} hr{luas.mean():>22,.0f} ha")
print("\nPemilihan sumber data harus sesuai jenis gangguan: deforestasi bertahap cukup dengan alert")
print("mingguan/bulanan, tetapi kebakaran yang cepat meluas membutuhkan data lebih sering (termal).")

print(f"\nFile di '{OUT}/':", sorted(os.listdir(OUT)))

# ====================================================================== TAMPILKAN HASIL (Colab / Jupyter)
judul("MENAMPILKAN HASIL")
try:
    notebook = get_ipython() is not None          # noqa: F821  (hanya ada di Colab/Jupyter)
except NameError:
    notebook = False

if notebook:
    from IPython.display import Image, display
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print(f"\n{f}")
            display(Image(f"{OUT}/{f}"))
else:
    print(f"Bukan notebook: semua gambar ada di folder '{OUT}/'.")

# 6 Studi Kasus dan Implementasi — Informatics Practitioner Lecture Series

Folder ini memisahkan notebook sumber menjadi **6 file Python**, satu file untuk setiap pertemuan. Isi kode studi kasus dipertahankan dari file notebook yang diberikan.

## Daftar file

1. `Pertemuan_1_Analitik_AI_PT_Maju_Bersama.py` — analitik dan AI untuk studi kasus perusahaan dengan data sintetis.
2. `Pertemuan_2_HPC_Simulasi_Evakuasi.py` — mini-LINPACK dan simulasi evakuasi; dapat dijalankan serial, MPI opsional.
3. `Pertemuan_3_GIS_Karir_dan_Persaingan.py` — studi kasus GIS: lokasi gerai, laporan warga, klasifikasi tutupan lahan, dan luas lahan.
4. `Pertemuan_4_Introduction_to_Machine_Learning.py` — kumpulan implementasi konsep machine learning.
5. `Pertemuan_5_IMK_dan_Visualisasi_Data.py` — studi kasus dashboard penjualan dan prinsip IMK/visualisasi.
6. `Pertemuan_6_AI_Pemantauan_Deforestasi.py` — simulasi pemantauan deforestasi dengan citra dan AI.

## Cara menjalankan

Disarankan menggunakan Python 3.10 atau 3.11 di virtual environment.

```bash
python -m venv .venv
```

Aktifkan environment:
- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Pasang dependensi umum:

```bash
pip install -r requirements.txt
```

Lalu jalankan salah satu file, misalnya:

```bash
python Pertemuan_1_Analitik_AI_PT_Maju_Bersama.py
```

Catatan:
- Pertemuan 2 mendukung MPI secara opsional melalui `mpi4py`; untuk mode paralel, pasang MPI dan `mpi4py` terlebih dahulu. Uji dengan ukuran kecil sebelum memakai ukuran benchmark besar.
- Pertemuan 6 mencoba memakai PyTorch jika tersedia; bila tidak, perilaku mengikuti fallback yang ada di kode sumber. Jika perlu menjalankan bagian CNN, pasang PyTorch yang cocok dengan perangkat.
- Data pada studi kasus yang dibuat secara sintetis bukan data asli perusahaan atau data lapangan.
- Kode dipecah dari notebook yang diberikan; dependensi khusus tambahan mungkin diperlukan jika ingin mengaktifkan seluruh opsi opsional.

## Upload ke GitHub

1. Buat repository baru di GitHub, misalnya `studi-kasus-machine-learning`.
2. Ekstrak ZIP ini.
3. Buka folder hasil ekstrak, lalu jalankan:

```bash
git init
git add .
git commit -m "Tambah 6 studi kasus machine learning"
git branch -M main
git remote add origin https://github.com/USERNAME/studi-kasus-machine-learning.git
git push -u origin main
```

Ganti `USERNAME` dengan username GitHub sendiri. Jika repository dibuat dengan README dari GitHub, ikuti petunjuk GitHub untuk menyelaraskan repository lokal terlebih dahulu.

# -*- coding: utf-8 -*-
"""
SESI 2 - HPC: Bridging Global Infrastructure to Address Local Challenge (Dr. Worawan Diaz Carballo)

Studi kasus mengikuti keynote:
  (A) FLOPS dan LINPACK (HPL): mini-benchmark penyelesaian sistem linear padat Ax = b
      untuk memperkirakan GFLOPS mesin Anda (prinsip yang sama dengan peringkat Top500).
  (B) Use case "evacuation and city-scale simulations": simulasi evakuasi tsunami berbasis
      agen pada banyak SKENARIO (waktu peringatan, jumlah pintu shelter, waktu reaksi).
      Skenario dibagi ke banyak proses MPI -> pola HPC "bagi kerja, hitung paralel, kumpulkan".
  (C) Alur kerja praktis HPC (komentar di bagian bawah): SSH key, batch script, scheduler,
      ambil hasil, post-processing dan visualisasi lokal.

Jalankan serial :  python sesi2_hpc_evakuasi_linpack.py
Jalankan MPI    :  mpirun -n 4 python sesi2_hpc_evakuasi_linpack.py
Colab           :  !mpirun --allow-run-as-root --oversubscribe -n 4 python sesi2_hpc_evakuasi_linpack.py
Opsi            :  --penduduk 5000 --ulangan 20 --ukuran-linpack 2000
"""
import argparse
import itertools
import time

import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
except Exception:                                   # tanpa MPI -> mode serial
    comm, rank, size = None, 0, 1

ap = argparse.ArgumentParser()
ap.add_argument("--penduduk", type=int, default=5_000)
ap.add_argument("--ulangan", type=int, default=20, help="replikasi Monte Carlo per skenario")
ap.add_argument("--ukuran-linpack", type=int, default=2000)
args, _ = ap.parse_known_args()                     # abaikan argumen kernel Jupyter/Colab


# ============================================================ (A) mini-LINPACK
def mini_linpack(n):
    """Selesaikan Ax=b dengan LU (LAPACK). Operasi floating point ~ (2/3)n^3 + 2n^2."""
    rng = np.random.default_rng(0)
    A, b = rng.random((n, n)), rng.random(n)
    t0 = time.perf_counter()
    x = np.linalg.solve(A, b)
    dt = time.perf_counter() - t0
    resid = np.linalg.norm(A @ x - b) / (np.linalg.norm(A) * np.linalg.norm(x) * n * np.finfo(float).eps)
    flops = (2 / 3) * n ** 3 + 2 * n ** 2
    return flops / dt / 1e9, dt, resid


if rank == 0:
    gflops, dt, resid = mini_linpack(args.ukuran_linpack)
    print(f"[A] mini-LINPACK n={args.ukuran_linpack}: {gflops:,.1f} GFLOPS ({dt:.2f} dtk), "
          f"residu ternormalisasi={resid:.2f} (< ~16 dianggap valid pada HPL)")
    print("    Satu core laptop ~ puluhan GFLOPS; supercomputer Top500 mencapai exaFLOPS (10^18).")


# ============================================================ (B) evakuasi tsunami
def simulasi_evakuasi(n_orang, waktu_peringatan_mnt, n_pintu, reaksi_mnt, rng):
    """Satu replikasi: berapa persen penduduk masuk shelter sebelum gelombang tiba?

    - Penduduk tersebar 100-1.000 m dari shelter; kecepatan jalan ~ N(1,2; 0,3) m/s (min 0,4).
    - Setiap orang menunggu 'reaksi' (menit) sebelum bergerak.
    - Shelter punya pintu; tiap pintu meloloskan 120 orang/menit (antrean -> bottleneck).
    - Gelombang tiba pada waktu_peringatan_mnt setelah peringatan.
    """
    jarak = rng.uniform(100, 1000, n_orang)
    kec = np.clip(rng.normal(1.2, 0.3, n_orang), 0.4, None)
    reaksi = rng.exponential(reaksi_mnt, n_orang) * 60.0                     # detik
    tiba = np.sort(reaksi + jarak / kec)                                    # waktu tiba di pintu
    laju = n_pintu * 2.0                                                    # orang/detik (120/menit/pintu)
    k = np.arange(n_orang)
    masuk = k / laju + np.maximum.accumulate(tiba - k / laju)               # antrean FIFO (rekurens tertutup)
    return float(np.mean(masuk <= waktu_peringatan_mnt * 60.0))


skenario = list(itertools.product(
    [10, 15, 20, 30],          # waktu peringatan sebelum gelombang tiba (menit)
    [1, 2, 3, 4],              # jumlah pintu shelter
    [1, 3, 5],                 # rata-rata waktu reaksi warga (menit)
))

# bagi skenario ke semua rank (round-robin)
t0 = time.perf_counter()
saya = skenario[rank::size]
hasil_lokal = []
for j, (w, pintu, rea) in enumerate(saya):
    rng = np.random.default_rng(10_000 * (rank + 1) + j)                    # seed berbeda tiap rank
    selamat = [simulasi_evakuasi(args.penduduk, w, pintu, rea, rng) for _ in range(args.ulangan)]
    hasil_lokal.append((w, pintu, rea, float(np.mean(selamat))))
waktu_lokal = time.perf_counter() - t0

semua = comm.gather(hasil_lokal, root=0) if comm else [hasil_lokal]
waktu_maks = comm.reduce(waktu_lokal, op=MPI.MAX, root=0) if comm else waktu_lokal

# ============================================================ post-processing (rank 0)
if rank == 0:
    hasil = sorted(h for bagian in semua for h in bagian)
    print(f"\n[B] simulasi evakuasi tsunami: {len(skenario)} skenario x {args.ulangan} replikasi x "
          f"{args.penduduk:,} penduduk, {size} proses")
    print(f"    waktu komputasi (terlama antar-rank): {waktu_maks:.2f} detik")
    print("\n    persen penduduk berhasil masuk shelter (rata-rata replikasi):")
    print("    peringatan | pintu | reaksi(mnt) | berhasil")
    for w, p, r, s in hasil:
        if r == 3:                                                          # tampilkan satu kolom reaksi
            print(f"    {w:>7} mnt | {p:>5} | {r:>11} | {s:>7.1%}")
    terburuk = min(hasil, key=lambda h: h[3])
    print(f"\n    skenario terburuk: peringatan {terburuk[0]} mnt, {terburuk[1]} pintu, reaksi {terburuk[2]} mnt "
          f"-> {terburuk[3]:.1%}")
    aman = [h for h in hasil if h[3] >= 0.95]
    print(f"    skenario dengan >=95% selamat: {len(aman)} dari {len(hasil)} "
          f"(dasar perencanaan: tambah pintu/jalur dan percepat peringatan)")
    print("\n    Ukur speedup: jalankan dengan n=1,2,4,8 proses dan bandingkan waktu komputasi.")

# ============================================================ (C) ALUR KERJA HPC
# 1) ssh-keygen -t ed25519 ; daftarkan PUBLIC key di portal HPC
# 2) ssh user@hpc.contoh.ac.id            # login node (Linux, terminal)
# 3) scp sesi2_hpc_evakuasi_linpack.py job_evakuasi.slurm user@hpc.contoh.ac.id:~/proyek/
# 4) sbatch job_evakuasi.slurm ; squeue -u $USER
# 5) scp user@hpc.contoh.ac.id:~/proyek/evakuasi_*.out .   # ambil hasil, visualisasi lokal
#
# ----- job_evakuasi.slurm (sesuaikan partisi dan modul dengan pusat HPC Anda) -----
# #!/bin/bash
# #SBATCH --job-name=evakuasi
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-node=16          # jumlah proses MPI per node
# #SBATCH --time=00:30:00               # walltime
# #SBATCH --output=evakuasi_%j.out
# module load python openmpi
# srun python sesi2_hpc_evakuasi_linpack.py --penduduk 50000 --ulangan 50

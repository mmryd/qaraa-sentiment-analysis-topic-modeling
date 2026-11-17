# fungsi visualisasi sentimen, dipanggil dari notebook

import os

import matplotlib.pyplot as plt
import pandas as pd

LABEL_SENTIMEN = ['positif', 'negatif', 'netral']
WARNA_SENTIMEN = {'positif': '#2ecc71', 'negatif': '#e74c3c', 'netral': '#95a5a6'}

# kata kunci tiap fitur, dicocokkan dengan batas kata agar tidak mengenai kata lain
FITUR_KATA_KUNCI = {
    'Materi Belajar': r'\b(?:belajar|materi)\b',
    "Baca Al-Qur'an": r'\b(?:membaca|baca|alquran|mengaji)\b',
    'Sensor Suara': r'\b(?:suara|rekam|perekaman)\b',
    'Notifikasi Adzan': r'\b(?:adzan|azan|notifikasi|alarm)\b',
    'Layanan Premium': r'\b(?:premium|langganan|membayar|bayar|iklan)\b',
}


def hitung_fitur(df, fitur_dict=FITUR_KATA_KUNCI, kolom_teks='content_final'):
    # satu ulasan bisa masuk beberapa fitur, sehingga total antar fitur tidak saling lepas
    teks = df[kolom_teks].fillna('').astype(str)
    hasil = []
    for fitur, pola in fitur_dict.items():
        sub = df[teks.str.contains(pola, case=False, regex=True)]
        hitung = sub['label'].value_counts()
        hasil.append({'Fitur': fitur, **{k: int(hitung.get(k, 0)) for k in LABEL_SENTIMEN}})
    return pd.DataFrame(hasil)


def _simpan(fig, path):
    # simpan gambar hanya jika path diberikan
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')


def plot_distribusi(counts, path=None):
    warna = [WARNA_SENTIMEN[label] for label in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=warna)

    # tampilkan jumlah ulasan di atas setiap batang
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 20, int(y),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("Distribusi Sentimen Ulasan Aplikasi Qara'a", fontsize=14)
    ax.set_ylabel('Jumlah Ulasan')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    _simpan(fig, path)
    plt.show()


def plot_proporsi(counts, path=None):
    warna = [WARNA_SENTIMEN[label] for label in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=warna)
    ax.set_title('Persentase Proporsi Sentimen')

    _simpan(fig, path)
    plt.show()


def plot_fitur(df_grafik, path=None):
    ax = df_grafik.set_index('Fitur')[LABEL_SENTIMEN].plot(
        kind='bar', figsize=(12, 7), width=0.7,
        color=[WARNA_SENTIMEN[l] for l in LABEL_SENTIMEN],
        edgecolor='black', linewidth=0.5,
    )
    fig = ax.get_figure()

    ax.set_title("Distribusi Sentimen Berdasarkan Fitur Aplikasi Qara'a",
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Fitur Aplikasi', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Jumlah Ulasan', fontsize=13, fontweight='bold', labelpad=10)
    ax.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Kategori Sentimen', title_fontsize=12, fontsize=11, loc='upper right')

    # tampilkan angka di atas batang yang nilainya lebih dari 0
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=10,
                        xytext=(0, 3), textcoords='offset points')

    fig.tight_layout()
    _simpan(fig, path)
    plt.show()
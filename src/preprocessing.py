# pipeline preprocessing ulasan qara'a
# harus identik dengan notebook agar inference konsisten dengan data latih

import re
from pathlib import Path

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

FILE_SLANG = Path(__file__).resolve().parent.parent / 'data' / 'kamus' / 'colloquial-indonesian-lexicon.csv'

# variasi frasa yang mengandung tanda baca atau spasi, diproses sebelum tanda baca dihapus
VARIASI_FRASA = {
    "qara'a": "qaraa",
    "qoro'a": "qaraa",
    "al-qur'an": "quran",
    "al-quran": "quran",
    "alqur'an": "quran",
    "al qur'an": "quran",
    "al quran": "quran",
    "alquran": "quran",
    "muraja'ah": "murajaah",
    "non aktif": "tidak aktif",
}

# tambahan kata manual sesuai kebutuhan aplikasi
TAMBAHAN_MANUAL = {
    # typo dan singkatan umum aplikasi
    'apk': 'aplikasi', 'apps': 'aplikasi', 'app': 'aplikasi',
    'aaplikasi': 'aplikasi', 'aplikasinya': 'aplikasi',
    'min': 'admin', 'mimim': 'admin',
    'download': 'unduh', 'update': 'perbarui',
    'loading': 'muat', 'lemot': 'lambat',
    'login': 'masuk', 'sign': 'masuk',

    # konsistensi kata rusak
    'error': 'eror',
    'crash': 'rusak',

    # istilah agama dan arab khas qara'a
    'qara': 'qaraa', 'qoro': 'qaraa', 'qoroa': 'qaraa', 'kora': 'qaraa',
    'qur': 'quran', 'alquran': 'quran',
    'ngaji': 'mengaji',
    'insyaallah': 'insya allah', 'subhanallah': 'subhanallah',
    'alhamdulillah': 'alhamdulillah', 'barakallah': 'barakallah',
    'syukron': 'terimakasih',

    # variasi tidak
    'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'tdk': 'tidak',
    'tak': 'tidak', 'ngga': 'tidak', 'engga': 'tidak',
    'enggak': 'tidak', 'nggak': 'tidak', 'ndak': 'tidak', 'kagak': 'tidak',
    'ora': 'tidak',

    # singkatan umum
    'yg': 'yang', 'langgananku': 'langganan',
    'sy': 'saya', 'aku': 'saya', 'aq': 'saya', 'gw': 'saya', 'gue': 'saya',
    'bgt': 'banget', 'bngt': 'banget',
    'tp': 'tapi', 'tpi': 'tapi',
    'krn': 'karena', 'utk': 'untuk', 'dgn': 'dengan',
    'dr': 'dari', 'dlm': 'dalam',
    'jd': 'jadi', 'jdi': 'jadi',
    'dah': 'sudah', 'sdh': 'sudah', 'udh': 'sudah',
    'blm': 'belum',
    'bisa': 'dapat', 'bs': 'bisa',
    'kalo': 'kalau', 'kl': 'kalau',
    'cuma': 'hanya', 'cuman': 'hanya',
    'kayak': 'seperti', 'kek': 'seperti',
    'aja': 'saja',
    'ni': 'ini', 'tu': 'itu',
    'sm': 'sama', 'trus': 'terus',
    'd': 'di', 'hp': 'ponsel',
    'nyah': 'nya', 'pengen': 'ingin',

    # ungkapan perasaan
    'thanks': 'terimakasih', 'makasih': 'terimakasih',
    'good': 'bagus', 'best': 'terbaik', 'nice': 'bagus',
    'ok': 'oke', 'oke': 'oke',
    'siip': 'bagus', 'sip': 'bagus', 'bagis': 'bagus',
    'mantap': 'bagus', 'mantul': 'bagus', 'keren': 'bagus',

    # kata tunggal yang sebelumnya ada di cleaning, dipindah ke sini
    'qoro': 'qaraa', 'qoroa': 'qaraa', 'kora': 'qaraa',
    'unsubscribe': 'berhenti langganan', 'unsubs': 'berhenti langganan',
    'nonaktif': 'tidak aktif',
}

# perbaikan kata yang over-stemmed oleh sastrawi
KAMUS_FIX = {
    'aji': 'mengaji',       'ngaji': 'mengaji',
    'quran': 'alquran',     'baca': 'membaca',
    'hafal': 'menghafal',   'ajar': 'belajar',
    'moga': 'semoga',       'langgan': 'langganan',
    'bayar': 'membayar',    'nyata': 'ternyata',
    'tuju': 'setuju',       'arti': 'artinya',
    'banget': 'sangat',     'kali': 'sekali',
    'enggak': 'tidak',

    # perbaikan makna sesuai temuan manual
    'tampil': 'tampilan',
    'tarik': 'menarik',
    'rusa': 'rusak',
}

# kata penting yang tidak boleh dihapus karena mengubah makna sentimen
_EXCEPTIONS = {
    'tidak', 'bukan', 'jangan', 'belum', 'kurang', 'tak',
    'akan', 'sedang', 'sudah', 'telah',
    'sangat', 'sekali', 'paling', 'banget',
    'tapi', 'namun', 'tetapi', 'walau', 'meski', 'padahal',
}

# stopword tambahan khusus konteks aplikasi ini
_MORE_STOPWORDS = [
    'sih', 'nya', 'nih', 'tuh', 'dong', 'deh', 'kok', 'mah',
    'kak', 'kakak', 'gan', 'sis', 'bro', 'pak', 'bu', 'min', 'admin',
    'yaitu', 'yakni', 'atas', 'dari', 'ke', 'di', 'ini', 'itu',
    'aplikasi', 'user', 'pengguna', 'orang',
]


def _muat_kamus_alay():
    # kamus slang dari salinan lokal, lalu ditambah tambahan_manual (urutan sama dengan notebook)
    df_kamus = pd.read_csv(FILE_SLANG)
    kamus = dict(zip(df_kamus['slang'], df_kamus['formal']))
    kamus.update(TAMBAHAN_MANUAL)
    return kamus


KAMUS_ALAY = _muat_kamus_alay()

FINAL_STOPWORDS = set(
    [w for w in StopWordRemoverFactory().get_stop_words() if w not in _EXCEPTIONS] + _MORE_STOPWORDS
)

STEMMER = StemmerFactory().create_stemmer()


def tahap_cleaning(text):
    # ganti variasi frasa dengan bentuk baku (batas kata agar tidak mengenai kata lain)
    for asal, tujuan in VARIASI_FRASA.items():
        text = re.sub(r'\b' + re.escape(asal) + r'\b', tujuan, text)

    # hapus mention dan hashtag
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)

    # hapus url
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # ganti karakter newline dengan spasi
    text = text.replace('\n', ' ')

    # hapus angka, simbol, tanda baca, dan emoji
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # hapus huruf tunggal yang berdiri sendiri
    text = re.sub(r'\b[a-zA-Z]\b', '', text)

    # rapikan huruf yang berulang panjang
    text = re.sub(r'(.)\1+', r'\1\1', text)

    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tahap_normalization(text):
    return ' '.join(KAMUS_ALAY.get(w, w) for w in text.split())


def tahap_stopwords(text):
    return ' '.join(w for w in text.split() if w not in FINAL_STOPWORDS)


def tahap_stemming(text):
    return STEMMER.stem(text)


def manual_stemming(text):
    return ' '.join(KAMUS_FIX.get(w, w) for w in text.split())


def preprocess_teks(teks):
    # urutan harus sama persis dengan pipeline data latih di notebook
    if teks is None or (isinstance(teks, float) and pd.isna(teks)):
        return ''
    teks = str(teks).lower()           # case folding
    teks = tahap_cleaning(teks)
    teks = tahap_normalization(teks)
    teks = tahap_stopwords(teks)
    teks = tahap_stemming(teks)
    teks = manual_stemming(teks)
    return ' '.join(teks.split())
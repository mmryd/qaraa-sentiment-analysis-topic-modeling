# pipeline preprocessing ulasan qara'a
# harus identik dengan notebook agar inference konsisten dengan data latih

import re
from pathlib import Path

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

FILE_SLANG = Path(__file__).resolve().parent.parent / 'data' / 'kamus' / 'colloquial-indonesian-lexicon.csv'

# tambahan kata manual sesuai kebutuhan aplikasi
TAMBAHAN_MANUAL = {
    # typo dan singkatan umum aplikasi
    'apk': 'aplikasi', 'apps': 'aplikasi', 'app': 'aplikasi',
    'aaplikasi': 'aplikasi', 'aplikasinya': 'aplikasi',
    'min': 'admin', 'mimim': 'admin',
    'download': 'unduh', 'mendownload': 'unduh', 'update': 'perbarui',
    'loading': 'muat', 'lemot': 'lambat',
    'login': 'masuk', 'sign': 'masuk',

    # konsistensi kata rusak dan error
    'error': 'eror',
    'crash': 'rusak',

    # istilah agama dan arab khas qara'a
    'qara': 'qaraa', 'qoro': 'qaraa', 'qoroa': 'qaraa', 'kora': 'qaraa',
    'qur': 'quran', 'alquran': 'quran', 'aquran': 'quran',
    'ngaji': 'mengaji',
    'insyaallah': 'insyaallah',
    'masya': 'masyaallah', 'masyaa': 'masyaallah',
    'subhanallah': 'subhanallah',
    'alhamdulillah': 'alhamdulillah', 'alhamdulilah': 'alhamdulillah',
    'allhamdulillah': 'alhamdulillah',
    'barakallah': 'barakallah',
    'syukron': 'terimakasih',
    'mudahan': 'mudahan',

    # ejaan baku salat dan hadis
    'sholat': 'salat', 'shalat': 'salat', 'solat': 'salat',
    'hadist': 'hadis', 'hadits': 'hadis',

    # sufiks -nya yang sering muncul, dipetakan eksplisit agar tidak memotong kata lain
    'apknya': 'aplikasi', 'appnya': 'aplikasi', 'appsnya': 'aplikasi',
    'aplikasihnya': 'aplikasi', 'aplilasinya': 'aplikasi',
    'loadingnya': 'muat', 'downloadnya': 'unduh',
    'fiturnya': 'fitur', 'piturnya': 'fitur',
    'adminnya': 'admin', 'servernya': 'server', 'responnya': 'respon',
    'adzannya': 'adzan', 'sholatnya': 'salat', 'solatnya': 'salat',
    'ngajinya': 'mengaji', 'dzikirnya': 'dzikir', 'ustadznya': 'ustadz',
    'hadistnya': 'hadis', 'hadisnya': 'hadis',
    'smuanya': 'semua', 'dlmnya': 'dalam', 'jdinya': 'jadi',
    'latinnya': 'latin', 'rejekinya': 'rezeki', 'bugnya': 'bug',
    'pembelajaranya': 'pembelajaran',

    # ejaan dan typo dari sampel ulasan
    'trims': 'terimakasih', 'trima': 'terimakasih', 'mksh': 'terimakasih',
    'tapu': 'tapi', 'bangat': 'banget', 'dngn': 'dengan',
    'mhami': 'memahami', 'bermangfaat': 'bermanfaat', 'pringatan': 'peringatan',
    'kalok': 'kalau', 'klo': 'kalau', 'klu': 'kalau',
    'mlm': 'malam', 'slalu': 'selalu',

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
    'kl': 'kalau',
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
    'unsubscribe': 'berhenti langganan', 'unsubs': 'berhenti langganan',
    'nonaktif': 'tidak aktif',

    # typo dan ejaan dari sampel ulasan kedua
    'mengngaji': 'mengaji',
    'ustazah': 'ustadzah', 'ustazh': 'ustadzah', 'ustad': 'ustadz',
    'buaya': 'buka',
    'usernya': 'pengguna',
    'sya': 'saya',
    'dev': 'developer',
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
    # kata inggris dan artikel yang tersisa dari ulasan
    'the', 'is', 'for', 'and', 'of', 'on', 'with', 'to', 'in', 'al',
    'btw', 'nnya', 'annya',
]


def _muat_kamus_alay():
    # kamus slang dari salinan lokal, lalu ditambah tambahan_manual
    df_kamus = pd.read_csv(FILE_SLANG)
    kamus = dict(zip(df_kamus['slang'], df_kamus['formal']))
    kamus.update(TAMBAHAN_MANUAL)
    return kamus


KAMUS_ALAY = _muat_kamus_alay()

FINAL_STOPWORDS = set(
    [w for w in StopWordRemoverFactory().get_stop_words() if w not in _EXCEPTIONS] + _MORE_STOPWORDS
)

STEMMER = StemmerFactory().create_stemmer()


def normalisasi_quran(text):
    # samakan apostrof melengkung dan tanda serupa
    text = text.replace('’', "'").replace('‘', "'").replace('`', "'")
    # al-qur'an, al qur,an, qur-an, qur~an, alwur,an, kur,an, qur'annya, dan pemisah lain
    text = re.sub(r"\b(?:al)?[\W_]*(?:qur|wur|kur)[\W_]*an(?:nya)?\b", 'quran', text)
    # sisa "quran an" dan "quran annya" yang terpisah spasi
    text = re.sub(r"\bquran\s+a*n+ya?\b", 'quran', text)
    return text


def normalisasi_umum(text):
    # ejaan dan typo yang harus dikenali sebelum angka dan tanda baca dihapus
    text = re.sub(r"\balloh\b", 'allah', text)
    text = re.sub(r"\bmudah[\W_]*2?[\W_]*an\b", 'mudahan', text)
    text = re.sub(r"\b(?:wara|q?ara)[\W_]*a\b", 'qaraa', text)
    text = re.sub(r"\bta[\W_]*ala\b", 'taala', text)
    return text


# frasa yang harus menjadi satu token
VARIASI_FRASA = {
    "qara'a": "qaraa",
    "qoro'a": "qaraa",
    "muraja'ah": "murajaah",
    "non aktif": "tidak aktif",
    # frasa religius menjadi satu token
    "masya allah": "masyaallah",
    "masya-allah": "masyaallah",
    "masha allah": "masyaallah",
    "masha-allah": "masyaallah",
    "masyaa allah": "masyaallah",
    "insya allah": "insyaallah",
    "insya-allah": "insyaallah",
    "terima kasih": "terimakasih",
}


def satukan_frasa(text):
    # ganti frasa multi-kata menjadi satu token, dipanggil di cleaning dan setelah normalisasi kamus
    for asal, tujuan in VARIASI_FRASA.items():
        text = re.sub(r'\b' + re.escape(asal) + r'\b', tujuan, text)
    return text


def tahap_cleaning(text):
    text = normalisasi_quran(text)
    text = normalisasi_umum(text)
    text = satukan_frasa(text)

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
    hasil = ' '.join(KAMUS_ALAY.get(w, w) for w in text.split())
    # kamus bisa menghasilkan frasa baru, misalnya "trims" menjadi "terima kasih"
    return satukan_frasa(hasil)


def tahap_stopwords(text):
    return ' '.join(w for w in text.split() if w not in FINAL_STOPWORDS)


def tahap_stemming(text):
    return STEMMER.stem(text)


def manual_stemming(text):
    return ' '.join(KAMUS_FIX.get(w, w) for w in text.split())


def preprocess_teks(teks):
    # urutan harus sama persis dengan pipeline data latih
    if teks is None or (isinstance(teks, float) and pd.isna(teks)):
        return ''
    teks = str(teks).lower()           # case folding
    teks = tahap_cleaning(teks)
    teks = tahap_normalization(teks)
    teks = tahap_stopwords(teks)
    teks = tahap_stemming(teks)
    teks = manual_stemming(teks)
    return ' '.join(teks.split())

def tahap_tokenizing(text):
    # pecah teks menjadi daftar kata, sama dengan tahap tokenizing di notebook
    return text.split()


def jalankan_tahap(df_raw):
    # jalankan seluruh tahap pada data mentah, kembalikan tabel tiap tahap dan log data yang dibuang
    # id diambil dari posisi baris raw, jangan diubah agar sejajar dengan validasi manual
    df = df_raw[['content', 'score']].copy()
    df.insert(0, 'id', range(len(df)))
    log = []

    df['content_cf'] = df['content'].str.lower()
    tahap = {'casefolding': df[['id', 'content', 'content_cf', 'score']].copy()}

    df['content_cleaning'] = df['content_cf'].apply(tahap_cleaning)
    kosong = df['content_cleaning'].str.strip() == ''
    log.append(df.loc[kosong, ['id', 'content', 'score']].assign(tahap='cleaning'))
    df = df[~kosong].reset_index(drop=True)
    tahap['cleaning'] = df[['id', 'content', 'content_cleaning', 'score']].copy()

    df['content_norm'] = df['content_cleaning'].apply(tahap_normalization)
    tahap['normalisasi'] = df[['id', 'content', 'content_norm', 'score']].copy()

    df['content_sw'] = df['content_norm'].apply(tahap_stopwords)
    kosong = df['content_sw'].str.strip() == ''
    log.append(df.loc[kosong, ['id', 'content', 'score']].assign(tahap='stopword'))
    df = df[~kosong].reset_index(drop=True)
    tahap['stopword'] = df[['id', 'content', 'content_sw', 'score']].copy()

    df['content_stem'] = df['content_sw'].apply(tahap_stemming).apply(manual_stemming)
    tahap['stemming'] = df[['id', 'content', 'content_stem', 'score']].copy()

    df['content_token'] = df['content_stem'].apply(tahap_tokenizing)
    tahap['tokenizing'] = df[['id', 'content', 'content_token', 'score']].copy()

    log_all = pd.concat(log, ignore_index=True)
    return tahap, log_all


def simpan_tahap(tahap, log_all, path_tahap, path_log):
    # tulis tiap tahap dan log ke path yang diberikan, dipanggil hanya jika flag pipeline aktif
    for nama, tabel in tahap.items():
        tabel.to_csv(path_tahap[nama], index=False, encoding='utf-8-sig')
    log_all.to_csv(path_log, index=False, encoding='utf-8-sig')
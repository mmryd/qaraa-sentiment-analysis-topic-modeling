# muat model dan lakukan prediksi dari teks mentah

from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import preprocess_teks

FILE_MODEL = Path(__file__).resolve().parent.parent / 'models' / 'model_svm_terbaik.joblib'


def muat_model(path=FILE_MODEL):
    # bundle berisi pipeline (tf-idf + svm), skema, C, dan daftar kelas
    return joblib.load(path)


def prediksi(teks, bundle):
    bersih = preprocess_teks(teks)
    if not bersih:
        return None

    model = bundle['model']
    kelas = bundle['kelas']

    # skor keputusan svm, label diambil dari skor tertinggi (sama dengan model.predict)
    skor = model.decision_function([bersih])[0]
    idx = int(np.argmax(skor))

    # softmax hanya untuk menampilkan skor relatif, bukan probabilitas terkalibrasi
    e = np.exp(skor - skor.max())
    relatif = e / e.sum()

    return {
        'label': kelas[idx],
        'skor_relatif': round(float(relatif[idx]) * 100, 1),
        'semua': {k: round(float(p) * 100, 1) for k, p in zip(kelas, relatif)},
        'skor_mentah': {k: round(float(s), 3) for k, s in zip(kelas, skor)},
        'teks_bersih': bersih,
    }
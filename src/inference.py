# muat model dan lakukan prediksi dari teks mentah

from pathlib import Path

import joblib

from src.preprocessing import preprocess_teks

FILE_MODEL = Path(__file__).resolve().parent.parent / 'models' / 'model_svm_terbaik.joblib'


def muat_model(path=FILE_MODEL):
    # bundle berisi pipeline (tf-idf + svm terkalibrasi), skema, C, dan daftar kelas
    return joblib.load(path)


def prediksi(teks, bundle):
    bersih = preprocess_teks(teks)
    if not bersih:
        return None

    model = bundle['model']
    kelas = bundle['kelas']
    proba = model.predict_proba([bersih])[0]
    idx = proba.argmax()

    return {
        'label': kelas[idx],
        'keyakinan': round(float(proba[idx]) * 100, 1),
        'semua': {k: round(float(p) * 100, 1) for k, p in zip(kelas, proba)},
        'teks_bersih': bersih,
    }
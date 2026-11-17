# pelabelan otomatis, aturan kandidat validasi manual, dan penggabungan label manual

import ast
import re

from src.preprocessing import preprocess_teks

NILAI_LABEL = {'positif', 'negatif', 'netral'}
NILAI_EDIT_VALID = NILAI_LABEL | {'hapus'}

NEGASI = {'tidak', 'bukan', 'jangan', 'belum', 'tak'}

# daftar kata awal untuk menandai kandidat validasi, bisa ditambah setelah melihat hasil
KATA_NEGATIF = {'rusak', 'lambat', 'gagal', 'kecewa', 'mahal', 'jelek', 'parah',
                'susah', 'sulit', 'hilang', 'macet', 'buruk'}
KATA_POSITIF = {'bagus', 'mantap', 'puas', 'lancar', 'terbaik', 'bermanfaat',
                'membantu', 'mudah', 'lengkap', 'alhamdulillah'}
POLA_PREMIUM = r'premium|langganan|bayar|iklan'


def pelabelan_otomatis(score):
    # skor 1 dan 2 negatif, skor 3 netral, skor 4 dan 5 positif
    if score <= 2:
        return 'negatif'
    elif score == 3:
        return 'netral'
    return 'positif'


def _ke_token(nilai):
    # token bisa berupa string list (dari csv) atau list (dari memori)
    if isinstance(nilai, str):
        return ast.literal_eval(nilai)
    return list(nilai)


def ada_kata(tokens, kata_set):
    # cocokkan kata jika tidak didahului negasi (misal "tidak rusak" tidak dihitung)
    for i, t in enumerate(tokens):
        if t in kata_set and not (i > 0 and tokens[i - 1] in NEGASI):
            return True
    return False


def alasan_baris(score, label, tokens):
    alasan = []
    if score == 3:
        alasan.append('netral')
    if label == 'positif' and ada_kata(tokens, KATA_NEGATIF):
        alasan.append('positif_ada_kata_negatif')
    if label == 'negatif' and ada_kata(tokens, KATA_POSITIF):
        alasan.append('negatif_ada_kata_positif')
    if label == 'negatif' and re.search(POLA_PREMIUM, ' '.join(tokens)):
        alasan.append('negatif_fitur_premium')
    if len(tokens) <= 2:
        alasan.append('pendek')
    return '; '.join(alasan)


def buat_label_auto(df_tok):
    # label otomatis hanya bergantung pada skor
    out = df_tok[['id', 'content', 'content_token', 'score']].copy()
    out['label'] = out['score'].apply(pelabelan_otomatis)
    return out[['id', 'content', 'content_token', 'score', 'label']]


def buat_perlu_cek(df_label):
    # kandidat validasi manual berdasarkan aturan, kolom label_manual_baru dikosongkan untuk diisi
    alasan = [
        alasan_baris(r.score, r.label, _ke_token(r.content_token))
        for r in df_label.itertuples(index=False)
    ]
    out = df_label.assign(alasan=alasan)
    out = out[out['alasan'] != ''].copy()
    out['label_manual_baru'] = ''
    return out[['id', 'content', 'content_token', 'score', 'label', 'alasan', 'label_manual_baru']]


def gabung_label_manual(df_auto, df_perlu):
    # label manual yang terisi menggantikan label otomatis berdasarkan id
    # baris yang kosong mempertahankan label otomatis
    edit = df_perlu[df_perlu['label_manual_baru'].notna()].copy()
    edit['label_manual_baru'] = edit['label_manual_baru'].str.strip().str.lower()

    tidak_valid = edit[~edit['label_manual_baru'].isin(NILAI_EDIT_VALID)]
    assert tidak_valid.empty, f"nilai label tidak valid:\n{tidak_valid[['id', 'label_manual_baru']]}"

    # jumlah perubahan dihitung terhadap label otomatis
    berubah = int((edit['label_manual_baru'] != edit['id'].map(df_auto.set_index('id')['label'])).sum())

    gab = df_auto.merge(edit[['id', 'label_manual_baru']], on='id', how='left', validate='one_to_one')
    gab['label'] = gab['label_manual_baru'].fillna(gab['label'])

    hapus = gab['label'] == 'hapus'
    log_hapus = gab.loc[hapus, ['id', 'content', 'score']].assign(tahap='validasi_manual')
    gab = gab[~hapus].drop(columns='label_manual_baru').reset_index(drop=True)
    return gab, log_hapus, berubah


def validasi_label(df):
    assert df['id'].is_unique, "ada id yang terduplikasi"
    assert df['label'].isin(NILAI_LABEL).all(), "ada label yang tidak valid"
    return df


def bangun_final(df_label, raw):
    # content_final dibangun dari teks mentah dengan preprocessing terbaru, label mengikuti validasi
    raw_id = raw[['content']].copy()
    raw_id.insert(0, 'id', range(len(raw_id)))

    df = df_label[['id', 'label', 'score']].merge(raw_id, on='id', how='left', validate='one_to_one')
    df['content_final'] = df['content'].apply(preprocess_teks)

    kosong = df['content_final'].str.strip() == ''
    log_final = df.loc[kosong, ['id', 'content', 'label', 'score']].assign(tahap='finalisasi')
    final = df[~kosong].reset_index(drop=True)[['id', 'content', 'content_final', 'label', 'score']]
    return final, log_final
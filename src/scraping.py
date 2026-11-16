# fungsi pengambilan ulasan google play, tidak dipanggil otomatis dari modul
from datetime import datetime

import pandas as pd
from google_play_scraper import reviews_all, Sort

def ambil_ulasan(app_id, start_date_str, end_date_str, lang='id', country='id'):
    # ambil seluruh ulasan terbaru lalu saring berdasarkan rentang tanggal
    result = reviews_all(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
    )
    df = pd.DataFrame(result)
    df['at'] = pd.to_datetime(df['at'])

    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')
    df = df[(df['at'] >= start) & (df['at'] <= end)]

    return df.reset_index(drop=True)
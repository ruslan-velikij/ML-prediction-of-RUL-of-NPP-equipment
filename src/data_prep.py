import os
import pandas as pd
import numpy as np


def prep_stat():
    """
    Читает сырые данные из datasets/raw/generic_pm/, для каждого device:
    – считает duration (в днях) до отказа/цензурирования,
    – флаг event (1=отказ, 0=не отказ),
    – извлекает baseline-признаки metric1–metric9 (из первой записи).
    Возвращает DataFrame с колонками:
    ['device','duration','event','metric1',...,'metric9']
    и сохраняет его в datasets/processed/statistical/stat_data.csv.
    """
    raw_folder = os.path.join('datasets', 'raw', 'generic_pm')
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in datasets/raw/generic_pm/"
            )

    df = pd.concat(
        [pd.read_csv(os.path.join(raw_folder, f)) for f in csv_files],
        ignore_index=True
    )
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

    records = []
    metrics_cols = [f'metric{i}' for i in range(1, 10)]

    for device, g in df.groupby('device'):
        g = g.sort_values('date')
        start = g['date'].iloc[0]

        event = int((g['failure'] == 1).any())
        fail_date = g.loc[g['failure'] == 1,
                          'date'].iloc[0] if event else g['date'].iloc[-1]

        raw_days = (fail_date - start).days
        duration = raw_days if raw_days > 0 else 1

        baseline = g.iloc[0][metrics_cols].to_dict()

        rec = {'device': device, 'duration': duration, 'event': event}
        rec.update(baseline)
        records.append(rec)

    stat_df = pd.DataFrame(records)
    processed_folder = os.path.join('datasets', 'processed', 'statistical')
    os.makedirs(processed_folder, exist_ok=True)
    stat_df.to_csv(os.path.join(
        processed_folder, 'stat_data.csv'), index=False
        )
    return stat_df


def prep_deg():
    """
    Читает CSV из datasets/raw/pwr_anomaly/PWR_Abnormality_Dataset.csv (без parse_dates).
    Вычисляет два кандидата:
      - RMS-вибраций по всем колонкам, начинающимся на 'VR'
      - Давление из колонки 'Pressure'
    Выбирает тот параметр, у которого более линейный (по R²) рост/дрейф.
    В качестве времени берёт просто порядковый номер измерения.
    Сохраняет временной ряд выбранного параметра в datasets/processed/degradation/deg_data.csv
    и возвращает DataFrame с колонками ['time', <best_param>].
    """
    pwr_file = 'datasets/raw/pwr_anomaly/PWR_Abnormality_Dataset.csv'
    if not os.path.isfile(pwr_file):
        raise FileNotFoundError(f"Файл не найден: {pwr_file}")

    # Чтение файла
    df = pd.read_csv(pwr_file)
    cols = df.columns.tolist()

    # Время — просто индекс строки
    df['time'] = np.arange(len(df))

    # 1) RMS всех вибрационных каналов (VR*)
    vib_cols = [c for c in cols if c.startswith('VR')]
    if not vib_cols:
        raise RuntimeError(f"Не найдено вибрационных колонок (VR*). Есть: {cols}")
    df['RMS_VIB'] = np.sqrt((df[vib_cols]**2).mean(axis=1))

    # 2) Давление
    if 'Pressure' not in df.columns:
        raise RuntimeError(f"Не найдена колонка 'Pressure'. Есть: {cols}")
    # Именование параметра для дальнейшего выбора
    df.rename(columns={'Pressure': 'PRESS'}, inplace=True)

    # Сравнение линейной аппроксимации для двух кандидатов
    candidates = ['RMS_VIB', 'PRESS']
    best_R2, best_param = -1.0, None
    x = df['time'].values
    for param in candidates:
        y = df[param].values
        if y.std() == 0:
            continue
        # y = m*x + c
        m, c = np.polyfit(x, y, 1)
        y_pred = m*x + c
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        if R2 > best_R2:
            best_R2, best_param = R2, param

    if best_param is None:
        raise RuntimeError("Ни один параметр не показал ненулевой дисперсии")

    # Подготовка итогового DataFrame и его сохранение
    deg_df = df[['time', best_param]].copy()
    out_dir = os.path.join('datasets', 'processed', 'degradation')
    os.makedirs(out_dir, exist_ok=True)
    deg_df.to_csv(os.path.join(out_dir, 'deg_data.csv'), index=False)

    return deg_df
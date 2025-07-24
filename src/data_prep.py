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
    и сохраняет его в datasets/processed/statistical/stat_data_D1.csv.
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
        processed_folder, 'stat_data_D1.csv'), index=False
        )
    return stat_df


def prep_deg():

    # 1. Чтение исходного CSV без парсинга дат:
    df = pd.read_csv('datasets/raw/pwr_anomaly/PWR_Abnormality_Dataset.csv')
    df['time'] = np.arange(len(df))

    # 2. Рассчёт RMS вибрации по столбцам VR*:
    vib_cols = [c for c in df.columns if c.startswith('VR')]
    df['RMS_VIB'] = np.sqrt((df[vib_cols]**2).mean(axis=1))

    df.rename(columns={'Pressure': 'PRESS'}, inplace=True)
    # 3. Выбор параметра с наибольшим R^2 линейного тренда:
    candidates = ['RMS_VIB', 'PRESS']
    best_param, best_R2 = None, -1
    x = df['time'].values
    for param in candidates:
        y = df[param].values
        if y.std() == 0:
            continue
        m, c = np.polyfit(x, y, 1)

        y_pred = m*x + c
        R2 = 1 - ((y - y_pred)**2).sum() / ((y - y.mean())**2).sum()
        if R2 > best_R2:
            best_R2 = R2
            best_param = param

    deg_df = df[['time', best_param]].copy()
    deg_df.to_csv('datasets/processed/degradation/deg_data_D2.csv', index=False)
    return deg_df

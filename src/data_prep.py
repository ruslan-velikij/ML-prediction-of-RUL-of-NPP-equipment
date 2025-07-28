import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


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


def _trend_coef(x):
    try:
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    except np.linalg.LinAlgError:
        return 0.0


def prep_reg():
    # 1. Чтение исходного CSV и удаление служебного индекса
    df = pd.read_csv('datasets/raw/pump_pm/rul_hrs.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # 2. Обработка временной метки
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    # 3. Заполнение пропусков медианой по каждому датчику
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    imputer = SimpleImputer(strategy='median')
    df[sensor_cols] = imputer.fit_transform(df[sensor_cols])

    # 4. Почасовая агрегация данных (усреднение сенсоров, последний RUL)
    agg_dict = {col: 'mean' for col in sensor_cols}
    agg_dict['rul'] = 'last'
    df_hourly = df.resample('h').agg(agg_dict)

    # 5. Извлечение дополнительных временных признаков
    df_hourly['month'] = df_hourly.index.month
    df_hourly['day'] = df_hourly.index.day
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
    df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)

    # 6. Скользящие агрегаты и тренд для каждого окна — соберём их в список
    feature_dfs = []  # сюда будем класть все новые блоки признаков
    window_sizes = [3, 6, 12, 24]

    for w in window_sizes:
        rolled = df_hourly[sensor_cols].rolling(window=w, min_periods=1)
        # базовые статистики
        feature_dfs.append(rolled.mean().add_suffix(f'_win{w}_mean'))
        feature_dfs.append(rolled.std().add_suffix(f'_win{w}_std'))
        feature_dfs.append(rolled.min().add_suffix(f'_win{w}_min'))
        feature_dfs.append(rolled.max().add_suffix(f'_win{w}_max'))
        feature_dfs.append(rolled.quantile(0.25).add_suffix(f'_win{w}_q25'))
        feature_dfs.append(rolled.quantile(0.75).add_suffix(f'_win{w}_q75'))
        # размах
        range_df = rolled.max().subtract(rolled.min()).add_suffix(f'_win{w}_range')
        feature_dfs.append(range_df)

        # тренд с проверкой на SVD
        def _trend(x):
            try:
                return np.polyfit(np.arange(len(x)), x, 1)[0]
            except np.linalg.LinAlgError:
                return 0.0

    trend_df = (
        df_hourly[sensor_cols]
        .rolling(window=w, min_periods=w)
        .apply(_trend_coef, raw=True)
        .add_suffix(f'_win{w}_trend')
    )
    feature_dfs.append(trend_df)

    # После цикла объединяем все вместе
    df_hourly = pd.concat([df_hourly] + feature_dfs, axis=1)

    # 7. Лаговые признаки (1h, 3h, 6h)
    lag_dfs = []
    for lag in [1, 3, 6]:
        df_shifted = df_hourly[sensor_cols].shift(lag)
        df_shifted.columns = [f'{c}_lag{lag}' for c in sensor_cols]
        lag_dfs.append(df_shifted)
    df_hourly = pd.concat([df_hourly] + lag_dfs, axis=1)

    # 8. Удаление строк с NaN (оставшиеся после окон и лагов)
    df_final = df_hourly.dropna()

    # 9. Сохранение подготовленного датасета
    os.makedirs('datasets/processed/regression', exist_ok=True)
    df_final.to_csv('datasets/processed/regression/pump_reg.csv', index=True)

    return df_final


def prep_sim():
    """Подготовка данных для модели, основанной на поиске похожих траекторий: извлечение отрезков перед каждым отказом."""
    W = 1440

    # 4.1. Загрузка сырых данных с датчиков
    raw_path = "datasets/raw/pump_sensor/sensor.csv"
    df = pd.read_csv(raw_path, index_col=0, parse_dates=["timestamp"])

    # 4.2. Удаление неинформативных или дублирующих сенсоров
    drop_cols = ["sensor_15", "sensor_50", "sensor_23"]
    df.drop(columns=drop_cols + ["machine_status", "timestamp"], inplace=True, errors='ignore')

    # 4.3. Интерполяция и заполнение пропусков для каждого сенсора
    df.interpolate(method='linear', axis=0, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # 4.4 Поиск индексов отказов (строки со статусом BROKEN в исходных данных)
    status_col = pd.read_csv(raw_path, usecols=["machine_status"])["machine_status"]
    fail_indices = np.where(status_col == "BROKEN")[0]

    # 4.5. Вырезание последних W измерений перед каждым отказом
    trajectories = []
    for idx in fail_indices:
        start_idx = idx - W
        if start_idx < 0:
            continue
        traj_df = df.iloc[start_idx: idx]
        # 4.6. Стандартизация внутри каждой траектории (z-преобразование по каждому сенсору)
        traj_vals = traj_df.to_numpy(dtype=np.float32)

        mean = np.nanmean(traj_vals, axis=0)
        std = np.nanstd(traj_vals, axis=0)

        std[std == 0] = 1e-9
        traj_vals = (traj_vals - mean) / std
        trajectories.append(traj_vals)

    # 4.7. Сохранение траекторий в файл формата NPZ
    save_path = "datasets/processed/similarity/traj.npz"
    traj_dict = {f"traj_{i}": traj for i, traj in enumerate(trajectories)}
    np.savez_compressed(save_path, **traj_dict)
    print(f"Сохранение {len(trajectories)} траекторий в {save_path}")

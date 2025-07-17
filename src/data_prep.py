import os
import pandas as pd


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

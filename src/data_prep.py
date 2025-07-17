import os
import pandas as pd


def prep_stat():
    """
    Читает сырые данные из datasets/raw/generic_pm/,
    вычисляет время до отказа для каждого устройства
    и сохраняет результат в datasets/processed/statistical/.
    Возвращает DataFrame stat_df с колонками:
    device, duration (в днях), event (флаг отказа).
    """
    raw_folder = os.path.join('datasets', 'raw', 'generic_pm')
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in datasets/raw/generic_pm/"
            )

    df_list = []
    for csv_file in csv_files:
        file_path = os.path.join(raw_folder, csv_file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True)

    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

    records = []
    for device, group in data.groupby('device'):
        group = group.sort_values('date')
        start_date = group['date'].iloc[0]
        event_occurred = int((group['failure'] == 1).any())

        if event_occurred:
            fail_date = group[group['failure'] == 1]['date'].iloc[0]
        else:
            fail_date = group['date'].iloc[-1]

        raw_days = (fail_date - start_date).days
        duration_days = raw_days if raw_days > 0 else 1

        records.append({
            'device': device,
            'duration': duration_days,
            'event': event_occurred
        })

    stat_df = pd.DataFrame(records)

    processed_folder = os.path.join('datasets', 'processed', 'statistical')
    os.makedirs(processed_folder, exist_ok=True)
    stat_df.to_csv(os.path.join(processed_folder, 'stat_data.csv'),
                   index=False)
    return stat_df

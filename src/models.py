import os
import joblib
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tslearn.metrics import cdist_dtw
from tslearn.neighbors import KNeighborsTimeSeriesRegressor
from lifelines import WeibullAFTFitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def fit_weibull(stat_df):
    """
    Обучает Weibull AFT-модель с ковариатами metric1–metric9.
    Сохраняет модель и survival-plot.
    """
    duration_col = 'duration'
    event_col = 'event'
    covariates = [c for c in stat_df.columns if c.startswith('metric')]

    df_fit = stat_df[covariates + [duration_col, event_col]].copy()

    scaler = StandardScaler()
    df_fit[covariates] = scaler.fit_transform(df_fit[covariates])

    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(df_fit, duration_col=duration_col, event_col=event_col)

    os.makedirs('models', exist_ok=True)
    joblib.dump(aft, os.path.join('models', 'model_stat_D1.pkl'))

    avg_cov = df_fit[covariates].median().to_frame().T
    sf = aft.predict_survival_function(avg_cov)

    plt.figure(figsize=(6, 4))
    plt.plot(sf.index, sf.iloc[:, 0], label='Средний профиль')
    plt.xlabel('Время работы (дни)')
    plt.ylabel('Вероятность безотказной работы S(t)')
    plt.title('Weibull AFT с ковариатами — survival function')
    plt.legend()

    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'stat_performance_D1.png'))
    plt.close()

    return aft


def fit_degradation(deg_df):

    # Выделение столбца с признаками деградации
    param_col = [col for col in deg_df.columns if col != 'time'][0]
    x = deg_df['time'].values
    y = deg_df[param_col].values

    # Линейная аппроксимация
    m, c = np.polyfit(x, y, 1)
    y_lin = m*x + c
    ss_tot = ((y - y.mean())**2).sum()
    ss_res_lin = ((y - y_lin)**2).sum()
    R2_lin = 1 - ss_res_lin/ss_tot if ss_tot > 0 else 0

    # Экспоненциальная аппроксимация (если все y > 0)
    if (y <= 0).any():
        R2_exp = -1
    else:
        log_y = np.log(y)
        B, logA = np.polyfit(x, log_y, 1)
        A = np.exp(logA)
        y_exp = A * np.exp(B * x)
        ss_res_exp = ((y - y_exp)**2).sum()
        R2_exp = 1 - ss_res_exp/ss_tot if ss_tot > 0 else 0

    # Выбор лучшей модели
    if R2_exp > R2_lin:
        model_type = 'exponential'
        R2_best = R2_exp
        coeff = {'A': A, 'B': B}

        if 'RMS' in param_col or 'vib' in param_col.lower():
            threshold = 4.5
        elif 'DeltaT' in param_col or 'dt' in param_col.lower():
            threshold = 0.85 * y[0]
        else:
            threshold = 1.2 * y[-1]

        t_cross = np.log(threshold/A)/B if (B != 0 and threshold > 0) else float('inf')
    else:
        model_type = 'linear'
        R2_best = R2_lin
        coeff = {'slope': m, 'intercept': c}
        if 'RMS' in param_col or 'vib' in param_col.lower():
            threshold = 4.5
        elif 'DeltaT' in param_col or 'dt' in param_col.lower():
            threshold = 0.85 * y[0]
        else:
            threshold = 1.2 * y[-1]
        t_cross = (threshold - c)/m if m != 0 else float('inf')

    # Расчет RUL
    RUL = 0.0 if (not np.isfinite(t_cross) or t_cross < 0) else float(t_cross)

    model_info = {
        'param': param_col,
        'model': model_type,
        'coefficients': coeff,
        'R2': round(R2_best, 3),
        'threshold': threshold,
        'RUL_days': round(RUL, 3)
    }
    joblib.dump(model_info, 'models/model_deg_D2.pkl')

    # Построение и сохранение графика
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, 'o-', label='Данные')
    if model_type == 'linear':
        x_line = np.linspace(0, max(x.max(), RUL), 100)
        plt.plot(x_line, m*x_line + c, label=f'Линейный тренд (R²={R2_lin:.2f})')
    else:
        x_line = np.linspace(0, max(x.max(), RUL), 100)
        plt.plot(x_line, A*np.exp(B*x_line), label=f'Экспон. тренд (R²={R2_exp:.2f})')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Порог')
    if RUL > 0:
        plt.axvline(x=RUL, color='magenta', linestyle=':', label='Прогноз RUL')
        plt.xlabel('Время (дни)')
        plt.ylabel(param_col)
        plt.title('Экстраполяция деградационного параметра')
        plt.legend()
        plt.savefig('results/deg_performance_D2.png')
        plt.close()
    return model_info


def train_rf_regressor(reg_df):
    # Разделение на X и y
    X = reg_df.drop(columns=['rul'])
    y = reg_df['rul']

    # Разделение на 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Обучение Random Forest (100 деревьев)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка качества
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Validation RMSE: {rmse:.3f}')
    print(f'Validation MAE : {mae:.3f}')

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    with open('models/model_reg_D3.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Визуализация True vs Pred и важности признаков
    importances = model.feature_importances_
    feat_names = X.columns
    top_idx = importances.argsort()[-10:][::-1]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax_scatter = axes[0]
    ax_scatter.scatter(y_test, y_pred, alpha=0.5)
    ax_scatter.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    ax_scatter.set_xlabel('True RUL')
    ax_scatter.set_ylabel('Predicted RUL')
    ax_scatter.set_title('True vs Predicted RUL')

    ax_bar = axes[1]
    ax_bar.barh(range(10), importances[top_idx][::-1], align='center')

    fig.subplots_adjust(left=0.25)
    ax_bar.set_yticks(range(10))
    ax_bar.set_yticklabels(feat_names[top_idx][::-1], fontsize=9)
    ax_bar.set_xlabel('Feature Importance')
    ax_bar.set_title('Top 10 Features for RUL Regression')

    os.makedirs('results', exist_ok=True)
    fig.tight_layout()
    fig.savefig('results/reg_performance_D3.png', bbox_inches='tight')
    plt.close(fig)

    return model, rmse, mae


def predict_dtw_knn(sim_ds=None):
    """
    Прогнозирование остаточного ресурса (RUL) с использованием подхода на основе схожести траекторий (DTW + kNN).
    sim_ds: Необязательный аргумент — если передан заранее загруженный набор траекторий,
    он будет использован; иначе траектории загрузятся из файла.
    """
    W = 1440
    step = 60
    k_neighbors = 3

    # Загрузка обработанных траекторий (исторические случаи отказа)
    if sim_ds is None:
        data = np.load("datasets/processed/similarity/traj.npz", allow_pickle=True)
        traj_keys = sorted(data.files, key=lambda x: int(x.split('_')[1]))
        trajectories = [data[key] for key in traj_keys]
    else:
        trajectories = sim_ds

    # Выбираем одну траекторию как «текущую» (имитация онлайн-прогноза).
    # Исключаем её из обучения и пробуем предсказать её RUL.
    # Здесь берём последнюю траекторию как текущую.
    current_traj = trajectories[-1]
    historical_trajs = trajectories[:-1]

    # Генерация обучающей выборки из исторических траекторий
    train_samples = []
    train_labels = []
    for traj in historical_trajs:
        L = traj.shape[0]
        for cut in range(0, L, step):
            if cut == 0:
                RUL = 1
            else:
                RUL = cut
            if RUL > W:
                RUL = W
            if RUL == W:
                continue
            end_idx = L - (RUL - 1)
            sample = traj[: end_idx]
            sample_mean = sample.mean(axis=0)
            sample_std = sample.std(axis=0)
            sample_std[sample_std == 0] = 1e-9
            sample_norm = (sample - sample_mean) / sample_std
            train_samples.append(sample_norm.astype(np.float32))
            train_labels.append(RUL)
    train_samples = np.array(train_samples, dtype=object)
    train_labels = np.array(train_labels, dtype=np.float32)

    # Вычисление DTW-матрици расстояний между обучающими примерами
    print("Вычисление матрицы DTW-расстояний между обучающими траекториями...")
    D_train = cdist_dtw(train_samples)

    # Обучаем kNN-регрессор с предвычисленной матрицей расстояний
    knn = KNeighborsTimeSeriesRegressor(n_neighbors=k_neighbors, metric="precomputed")
    knn.fit(D_train, train_labels)
    print(f"kNN-модель обучена на {len(train_samples)} samples.")

    # Прогнозируем RUL для текущей траектории
    cur_mean = current_traj.mean(axis=0)
    cur_std = current_traj.std(axis=0)
    cur_std[cur_std == 0] = 1e-9
    current_norm = (current_traj - cur_mean) / cur_std

    D_query = cdist_dtw(train_samples, [current_norm])
    D_query = D_query.flatten().astype(np.float32)

    y_pred = knn.predict(D_query.reshape(1, -1))
    predicted_rul = float(y_pred[0])
    print(f"Прогнозируемый RUL для текущей траектории: ~{predicted_rul:.1f} minutes")
    nn_index = int(np.argmin(D_query))

    # 6. Построение графика: текущая траектория vs. лучшая историческая
    plt.figure(figsize=(8, 5))
    sensor_idx =  list(range(current_traj.shape[1])).index(22) if current_traj.shape[1] > 22 else 0
    plt.plot(current_traj[:, sensor_idx], label="Текущая траектория", color="blue")
    hist_traj_index = None
    cum_count = 0
    for t_index, traj in enumerate(historical_trajs):
        sample_count = (traj.shape[0] // step) + 1
        if nn_index < cum_count + sample_count:
            hist_traj_index = t_index
            break
        cum_count += sample_count
    if hist_traj_index is None:
        hist_traj_index = 0
    best_traj_full = historical_trajs[hist_traj_index]
    plt.plot(best_traj_full[:, sensor_idx], label="Похожая историческая траектория", color="orange", linestyle="--")
    cur_len = current_traj.shape[0]
    hist_len = best_traj_full.shape[0]
    plt.axvline(x=cur_len - 1, color="blue", linestyle=":")
    plt.text(cur_len, current_traj[-1, sensor_idx], f"  Прогнозируемый RUL ≈ {predicted_rul:.0f} min", color="blue",
             va='bottom', fontweight='bold')
    plt.axvline(x=hist_len - 1, color="orange", linestyle=":")
    plt.text(hist_len, best_traj_full[-1, sensor_idx], "  Отказ (исторический)", color="orange", va='bottom')
    plt.xlabel("Время (минуты деградации)")
    plt.ylabel("Значение сенсора 22 (норм.)")
    plt.title("Сравнение текущей и исторической траекторий")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/sim_performance_D4.png")
    plt.close

    # Сохраняем модель и данные (DTW-матрицу и т.д.) для повторного использования
    model_data = {
        "dtw_matrix": D_train.astype(np.float32),
        "rul_train": train_labels.astype(np.float32),
        "knn": knn
    }
    import pickle
    with open("models/model_sim_D4.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Модель и данные сохранены в models/model_sim_D4.pkl")

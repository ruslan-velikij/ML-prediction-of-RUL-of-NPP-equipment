import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler


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
    joblib.dump(aft, os.path.join('models', 'model_stat.pkl'))

    avg_cov = df_fit[covariates].median().to_frame().T
    sf = aft.predict_survival_function(avg_cov)

    plt.figure(figsize=(6, 4))
    plt.plot(sf.index, sf.iloc[:, 0], label='Средний профиль')
    plt.xlabel('Время работы (дни)')
    plt.ylabel('Вероятность безотказной работы S(t)')
    plt.title('Weibull AFT с ковариатами — survival function')
    plt.legend()

    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'stat_performance.png'))
    plt.close()

    return aft


def fit_degradation(deg_df):
    """
    Строит линейный и экспоненциальный тренды по деградационному параметру.
    Выбирает лучший по R², вычисляет время до порогового значения (RUL).
    Сохраняет параметры выбранного тренда в models/model_deg.pkl
    и график с трендом и порогом в results/deg_performance.png.
    """
    # Определение столбца параметра (помимо 'time')
    param_col = [col for col in deg_df.columns if col != 'time'][0]
    x = deg_df['time'].values  # время (в днях)
    y = deg_df[param_col].values
    # Линейная модель: y = m*x + c
    m, c = np.polyfit(x, y, 1)
    y_lin_pred = m * x + c
    ss_res_lin = ((y - y_lin_pred)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    R2_lin = 1 - ss_res_lin/ss_tot if ss_tot > 0 else 0
    # Экспоненциальная модель: y = A * exp(B*x). Применение логарифма для линейной регрессии.
    if (y <= 0).any():
        R2_exp = -1
    else:
        log_y = np.log(y)
        B, logA = np.polyfit(x, log_y, 1)
        A = np.exp(logA)
        y_exp_pred = A * np.exp(B * x)
        ss_res_exp = ((y - y_exp_pred)**2).sum()
        R2_exp = 1 - ss_res_exp/ss_tot if ss_tot > 0 else 0
    # Выбор модели с большим R²
    if R2_exp > R2_lin:
        model_type = 'exponential'
        # Параметры экспоненциальной модели
        coeff = {'A': A, 'B': B}
        R2_best = R2_exp
        # Вычисление времени до порога для экспоненты: y = A * exp(B*t) достигнет threshold
        # Определение порогового значения на основе типа параметра:
        if 'RMS' in param_col or 'vib' in param_col.lower():
            threshold = 4.5  # критический уровень вибрации (мм/с) по ISO 10816
        elif 'DeltaT' in param_col or 'dt' in param_col.lower():
            threshold = 0.85 * y[0]  # допустимое снижение ΔT на 15%
        else:
            # Для газов или прочих параметров берем 20% рост от текущего последнего значения
            threshold = 1.2 * y[-1]
        # Решить уравнение A * exp(B*t) = threshold -> t = (ln(threshold/A)) / B
        if B != 0 and threshold > 0:
            t_cross = np.log(threshold / A) / B
        else:
            t_cross = float('inf')
    else:
        model_type = 'linear'
        # Параметры линейной модели
        coeff = {'slope': m, 'intercept': c}
        R2_best = R2_lin
        # Пороговое значение для линейного тренда
        if 'RMS' in param_col or 'vib' in param_col.lower():
            threshold = 4.5
        elif 'DeltaT' in param_col or 'dt' in param_col.lower():
            threshold = 0.85 * y[0]
        else:
            threshold = 1.2 * y[-1]
        if m != 0:
            t_cross = (threshold - c) / m
        else:
            t_cross = float('inf')
    # Если расчет дал отрицательное или бесконечное время (порог уже достигнут или недостижим), приравниваем RUL к 0
    if not np.isfinite(t_cross) or t_cross < 0:
        RUL_days = 0.0
    else:
        RUL_days = float(t_cross)
    # Сохранение выбранной модели и параметров
    model_info = {
        'param': param_col,
        'model': model_type,
        'coefficients': coeff,
        'R2': round(R2_best, 3),
        'threshold': threshold,
        'RUL_days': round(RUL_days, 3)
    }
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_info, os.path.join('models', 'model_deg.pkl'))
    # Построение графика тренда
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, 'o-', label='Данные')
    # Лучшая аппроксимация
    if model_type == 'linear':
        x_line = np.linspace(0, max(x.max(), RUL_days), 100)
        y_line = m * x_line + c
        plt.plot(x_line, y_line, label=f'Линейный тренд (R²={R2_lin:.2f})')
    else:
        x_line = np.linspace(0, max(x.max(), RUL_days), 100)
        y_line = A * np.exp(B * x_line)
        plt.plot(x_line, y_line, label=f'Экспоненциальный тренд (R²={R2_exp:.2f})')
    # Горизонтальная линия порога
    plt.axhline(y=threshold, color='red', linestyle='--', label='Порог')
    # Вертикальная линия до пересечения с порогом (если RUL > 0)
    if RUL_days > 0:
        plt.axvline(x=RUL_days, color='magenta', linestyle=':', label='Прогноз RUL')
    plt.xlabel('Время (дни)')
    plt.ylabel(param_col)
    plt.title('Экстраполяция деградационного параметра')
    plt.legend(loc='best')
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'deg_performance.png'))
    plt.close()
    return model_info

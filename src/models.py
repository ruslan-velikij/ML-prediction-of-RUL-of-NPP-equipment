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
        model_type = 'exponential';  R2_best = R2_exp
        coeff = {'A': A, 'B': B}

        if 'RMS' in param_col or 'vib' in param_col.lower():
            threshold = 4.5
        elif 'DeltaT' in param_col or 'dt' in param_col.lower():
            threshold = 0.85 * y[0]
        else:
            threshold = 1.2 * y[-1]

        t_cross = np.log(threshold/A)/B if (B != 0 and threshold > 0) else float('inf')
    else:
        model_type = 'linear';  R2_best = R2_lin
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
    plt.figure(figsize=(6,4))
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
    plt.xlabel('Время (дни)');  plt.ylabel(param_col)
    plt.title('Экстраполяция деградационного параметра')
    plt.legend()
    plt.savefig('results/deg_performance_D2.png');  plt.close()
    return model_info

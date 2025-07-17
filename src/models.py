import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
# Импортируем WeibullFitter из lifelines для построения Weibull-модели
from lifelines import WeibullFitter


def fit_weibull(stat_df):
    """
    Подгоняет Weibull AFT-модель на основе данных stat_df (время до отказа).
    Возвращает обученный объект модели, сохраняет его в models/model_stat.pkl.
    Также строит график функции выживания с отметкой текущего возраста
    и сохраняет в results/stat_performance.png.
    """
    durations = stat_df['duration']
    events = stat_df['event']

    wbf = WeibullFitter()
    wbf.fit(durations, events)

    joblib.dump(wbf, os.path.join('models', 'model_stat.pkl'))

    plt.figure(figsize=(6, 4))
    ax = wbf.plot_survival_function(ci_show=False)

    current_age = wbf.median_survival_time_
    survival_at_current = float(
        np.exp(-((current_age / wbf.lambda_) ** wbf.rho_))
        )
    ax.axvline(current_age, color='red', linestyle='--',
               label=f'Текущий возраст ≈ {current_age:.0f} дней')
    ax.axhline(survival_at_current, color='red', linestyle='--')
    ax.scatter(current_age, survival_at_current, color='red')

    plt.xlabel('Время работы оборудования (дни)')
    plt.ylabel('Вероятность безотказной работы (S(t))')
    plt.title('Weibull модель — функция выживания')
    plt.legend(loc='best')

    results_path = os.path.join('results', 'stat_performance.png')
    plt.savefig(results_path)
    plt.close()

    return wbf

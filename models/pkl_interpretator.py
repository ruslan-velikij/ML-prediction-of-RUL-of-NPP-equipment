import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
# 1.1) Weibull AFT model. Прокладка путей
    print("\n=== Weibull AFT model summary ===")
    processed_csv = os.path.join('datasets', 'processed',
                                 'statistical', 'stat_data_D1.csv')
    model_pkl = os.path.join('models', 'model_stat_D1.pkl')
    os.makedirs('results', exist_ok=True)

    # 1.2) Загрузка данных и моделей
    df = pd.read_csv(processed_csv)
    aft = joblib.load(model_pkl)

    # 1.3) Определение ковариатов для статистической модели
    covs = [c for c in df.columns if c.startswith('metric')]

    scaler = StandardScaler()
    df[covs] = scaler.fit_transform(df[covs])

    # 1.4) Вывод summary Weibull AFT
    print(aft.summary)

    # 1.5) Бар-чарт коэффициентов (lambda_, metric1–9)
    coef_lambda = aft.summary.xs('lambda_',
                                 level='param')['coef'].reindex(covs)
    plt.figure(figsize=(8, 4))
    coef_lambda.plot.bar()
    plt.axhline(0, color='black', lw=0.8)
    plt.title("Коэффициенты Weibull AFT (lambda_, metric1–metric9)")
    plt.ylabel("coef")
    plt.tight_layout()
    plt.savefig('results/stat_feature_coefficients_D1.png')
    plt.close()
    print("→ results/stat_feature_coefficients_D1.png")

    # 1.6) Survival-кривые для трёх типовых профилей
    profiles = {
        '25-й перцентиль': df[covs].quantile(0.25),
        'медиана':         df[covs].median(),
        '75-й перцентиль': df[covs].quantile(0.75),
    }
    plt.figure(figsize=(6, 4))
    for name, vals in profiles.items():
        sf = aft.predict_survival_function(vals.to_frame().T)
        plt.plot(sf.index, sf.iloc[:, 0], label=name)

    plt.xlabel("Время работы (дни)")
    plt.ylabel("S(t) — вероятность безотказной работы")
    plt.title("Survival-кривые для разных сенсорных профилей")
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/stat_survival_profiles_D1.png')
    plt.close()
    print("→ results/stat_survival_profiles_D1.png")

# 2.1) Интерпретация модели деградации. Прокладка путей
    print("\n\n\n=== Degradation Model Summary ===")
    model_deg_pkl = os.path.join('models', 'model_deg_D2.pkl')
    deg = joblib.load(model_deg_pkl)

    #2.2) Вывод ключевых параметров модели деградации
    for key, val in deg.items():
        print(f"{key}: {val}")

    #2.3 Уточнение порога и прогноза RUL
    if 'threshold' in deg and 'RUL_days' in deg:
        print(f"Порог: {deg['threshold']}, прогноз RUL: {deg['RUL_days']} единиц времени")

    #2.4 Визуализация тренда деградационного параметра
    deg_data_csv = os.path.join('datasets', 'processed', 'degradation', 'deg_data_D2.csv')
    if os.path.exists(deg_data_csv):
        deg_df = pd.read_csv(deg_data_csv)
        x = deg_df['time'].values
        y = deg_df[deg['param']].values

        plt.figure(figsize=(6, 4))
        plt.plot(x, y, 'o-', label='Данные')
        if deg['model'] == 'linear':
            m = deg['coefficients']['slope']
            c = deg['coefficients']['intercept']
            plt.plot(x, m * x + c,
                     label=f"Линейная модель (R²={deg['R2']})")
        else:
            A = deg['coefficients']['A']
            B = deg['coefficients']['B']
            plt.plot(x, A * np.exp(B * x),
                     label=f"Экспон. модель (R²={deg['R2']})")

        plt.axhline(deg['threshold'], color='red', linestyle='--', label='Порог')
        if deg['RUL_days'] > 0:
            plt.axvline(deg['RUL_days'], color='magenta', linestyle=':', label='Прогноз RUL')

        plt.xlabel('time')
        plt.ylabel(deg['param'])
        plt.title('Интерпретация модели деградации')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/deg_performance_interpretation_D2.png')
        plt.close()
        print("→ results/deg_performance_interpretation_D2.png")
    else:
        print(f"Не найден файл с данными для деградации: {deg_data_csv}")


if __name__ == '__main__':
    main()

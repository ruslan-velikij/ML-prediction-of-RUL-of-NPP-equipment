import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

# 3.1) Интерпретация регрессионной модели RUL. Прокладка путей
    print("\n\n\n=== Regression Model Interpretation ===")
    reg_csv       = os.path.join('datasets', 'processed', 'regression', 'pump_reg.csv')
    reg_model_pkl = os.path.join('models', 'model_reg_D3.pkl')

    #3.2 Загрузка датасета
    df_reg = pd.read_csv(reg_csv, index_col=0)
    X = df_reg.drop(columns=['rul'])
    y = df_reg['rul']

    #3.3 Воспроизводимое разбиение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #3.4 Загрузка обученной модели
    model_reg = joblib.load(reg_model_pkl)

    #3.5 Предсказание и метрики
    y_train_pred = model_reg.predict(X_train)
    mse_train   = mean_squared_error(y_train, y_train_pred)
    rmse_train  = np.sqrt(mse_train)
    mae_train   = mean_absolute_error(y_train, y_train_pred)
    print(f"[TRAIN] RMSE: {rmse_train:.3f}, MAE: {mae_train:.3f}")

    y_test_pred = model_reg.predict(X_test)
    mse_test   = mean_squared_error(y_test, y_test_pred)
    rmse_test  = np.sqrt(mse_test)
    mae_test   = mean_absolute_error(y_test, y_test_pred)
    print(f"[TEST]  RMSE: {rmse_test:.3f}, MAE: {mae_test:.3f}")

    #3.6 Визуализация важности признаков (топ-10)
    importances = model_reg.feature_importances_
    feat_names  = X.columns
    top_idx     = importances.argsort()[::-1][:10]

    plt.figure(figsize=(8,4))
    plt.barh(
        np.arange(10),
        importances[top_idx][::-1],
        align='center'
    )
    plt.yticks(
        np.arange(10),
        feat_names[top_idx][::-1]
    )
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Features for RUL Regression")
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/reg_feature_importance_D3.png')
    plt.close()
    print("→ results/reg_feature_importance_D3.png")

    # график True vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    diag = [y_test.min(), y_test.max()]
    plt.plot(diag, diag, 'k--')
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL (Test)")
    plt.tight_layout()
    plt.savefig('results/reg_true_vs_pred_D3.png')
    plt.close()
    print("→ results/reg_true_vs_pred_D3.png")


if __name__ == '__main__':
    main()

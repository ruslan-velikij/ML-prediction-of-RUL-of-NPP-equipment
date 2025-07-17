import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def main():
    # 1) Прокладка пути
    processed_csv = os.path.join('datasets', 'processed',
                                 'statistical', 'stat_data.csv')
    model_pkl = os.path.join('models', 'model_stat.pkl')
    os.makedirs('results', exist_ok=True)

    # 2) Загрузка данных и модели
    df = pd.read_csv(processed_csv)
    aft = joblib.load(model_pkl)

    # 3) Определение ковариатов
    covs = [c for c in df.columns if c.startswith('metric')]

    scaler = StandardScaler()
    df[covs] = scaler.fit_transform(df[covs])

    # 4) Вывод summary
    print("\n=== Weibull AFT model summary ===")
    print(aft.summary)

    # 5) Бар-чарт коэффициентов (параметр lambda_, ковариаты metric1–9)
    coef_lambda = aft.summary.xs('lambda_',
                                 level='param')['coef'].reindex(covs)
    plt.figure(figsize=(8, 4))
    coef_lambda.plot.bar()
    plt.axhline(0, color='black', lw=0.8)
    plt.title("Коэффициенты Weibull AFT (lambda_, metric1–metric9)")
    plt.ylabel("coef")
    plt.tight_layout()
    plt.savefig('results/feature_coefficients.png')
    plt.close()
    print("→ results/feature_coefficients.png")

    # 6) Survival-кривые для трёх типовых профилей
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
    plt.savefig('results/survival_profiles.png')
    plt.close()
    print("→ results/survival_profiles.png")


if __name__ == '__main__':
    main()

import os
import joblib
import matplotlib.pyplot as plt
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

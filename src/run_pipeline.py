import sys
import os
from data_prep import prep_stat, prep_deg, prep_reg
from models import fit_weibull, fit_degradation, train_rf_regressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    """Entry point for the pipeline."""

    # ================= Модель 1: Weibull AFT regressor =================
    print("Подготовка данных для статистического анализа отказов...")
    stat_df = prep_stat()

    print("Обучение Weibull‑модель и построение survival‑кривой…")
    fit_weibull(stat_df)

    print(
        "Готово! Weibull‑модель сохранена в models/model_stat_D1.pkl, "
        "график — в results/stat_performance_D1.png",
    )

    # ========== Модель 2: Линейная/Экспоненциальная регрессия ==========
    print("\n\nПодготовка данных для модели экстраполяции деградации…")
    deg_df = prep_deg()

    print("Посроение линейной и экспоненциальной регрессии и вычисление RUL…")
    model_info = fit_degradation(deg_df)

    print(
        f"Готово! Выбрана {model_info['model']}‑модель (R² = {model_info['R2']}), "
        f"прогноз RUL = {model_info['RUL_days']} единиц времени",
    )
    print(
        "Файл модели — models/model_deg_D2.pkl, "
        "график — results/deg_performance_D2.png",
    )

    # === Модель 3: RandomForestRegressor / GradientBoostingRegressor ===
    print("\n\nПодготовка данных для модели регрессии")
    reg_df = prep_reg()

    print("Обучение модели и оценка")
    model, rmse, mae = train_rf_regressor(reg_df)

    print('Pipeline completed.')
    print(f'Final RMSE: {rmse:.3f}, MAE: {mae:.3f}')


if __name__ == "__main__":
    main()

import sys
import os
from data_prep import prep_stat, prep_deg
from models import fit_weibull, fit_degradation

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    """Entry point for the pipeline."""

    # ================= Модель 1: Weibull AFT regressor =================
    print("Подготавливаем данные для статистического анализа отказов...")
    stat_df = prep_stat()

    print("Обучаем Weibull‑модель и строим survival‑кривую…")
    fit_weibull(stat_df)

    print(
        "Готово! Weibull‑модель сохранена в models/model_stat_D1.pkl, "
        "график — в results/stat_performance_D1.png",
    )

    # ========== Модель 2: Линейная/Экспоненциальная регрессия ==========
    print("\nПодготавливаем данные для модели экстраполяции деградации…")
    deg_df = prep_deg()

    print("Строим линейную и экспоненциальную регрессии и вычисляем RUL…")
    model_info = fit_degradation(deg_df)

    print(
        f"Готово! Выбрана {model_info['model']}‑модель (R² = {model_info['R2']}), "
        f"прогноз RUL = {model_info['RUL_days']} единиц времени",
    )
    print(
        "Файл модели — models/model_deg_D2.pkl, "
        "график — results/deg_performance_D2.png",
    )


if __name__ == "__main__":
    main()
